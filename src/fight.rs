// SPDX-License-Identifier: AGPL-3.0-only
// sword-sim by jo29_29. Licensed under the GNU AGPL v3 only with additional terms: see LICENSE and ADDITIONAL_TERMS.
//! A sword fight on the 1.21.11 sim (world.rs, server.rs, client.rs): two clients, each with its own ping, on
//! the sword arena with the sword kit (diamond armor Protection III + Unbreaking III, diamond sword Unbreaking III).
//!
//! A fight ends when a player dies (the other wins), leaves the arena, or at the time limit (a draw).
//!
//! The ping: each side has a base delay up and down in whole ticks; every tick it may wander by one tick from the
//! base (jitter), and now and then it spikes for a while. A packet keeps its place behind the ones sent before it,
//! as on TCP, so a spike holds back everything after it.
//!
//! What a client could know about its opponent but the sim's client does not keep (health, the hurt and swing
//! animations) comes from the server as it was `down` ticks ago.
use std::collections::VecDeque;
use std::sync::OnceLock;

use crate::sim::client::*;
use crate::sim::server::*;
use crate::sim::world::*;

/// splitmix64.
#[derive(Clone)]
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// lo..=hi.
    pub fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next_u64() % (hi - lo + 1) as u64) as i64
    }
    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }
}

// ------------------------------------------------------------------ the kit

pub fn kit_ench(item: Item) -> Ench {
    let mut e = Ench::default();
    match item {
        // Protection III, Unbreaking III armor; Unbreaking III sword.
        Item::DHelmet | Item::DChestplate | Item::DLeggings | Item::DBoots => {
            e.protection = 3;
            e.unbreaking = 3;
        }
        Item::DSword => {
            e.unbreaking = 3;
        }
        _ => {}
    }
    e
}

pub fn stack(item: Item, count: i32) -> Stack {
    Stack { item, count, damage: 0, ench: kit_ench(item) }
}

/// The sword kit: diamond armor and a diamond sword in hotbar slot 0, nothing else. Inventory, armor (feet, legs,
/// chest, head), offhand.
pub fn sword_kit() -> ([Stack; 36], [Stack; 4], Stack) {
    let mut inv = [Stack::EMPTY; 36];
    inv[0] = stack(Item::DSword, 1);
    let armor = [stack(Item::DBoots, 1), stack(Item::DLeggings, 1), stack(Item::DChestplate, 1), stack(Item::DHelmet, 1)];
    (inv, armor, Stack::EMPTY)
}

// ------------------------------------------------------------------ the arena

/// The level's height.
pub const HEIGHT: i32 = 32;

/// The sword arena (src/arena.txt): an oval floor out to x = +-SWORD_HALF and z = +-SWORD_HALF_Z (67 by 65), walled all round; block (0, 0) is its centre.
pub const SWORD_HALF: i32 = 33;
pub const SWORD_HALF_Z: i32 = 32;
/// The fixed start: blocks x = +-SWORD_SPAWN_X, z = 0, facing each other.
pub const SWORD_SPAWN_X: i32 = 33;

struct SwordMap {
    /// Per cell of the box (x -SWORD_HALF..=SWORD_HALF, z the same with SWORD_HALF_Z): floor or not.
    floor: Vec<bool>,
    /// Per cell: the wall cell nearest its centre (cells outside the box are wall).
    near: Vec<(i32, i32)>,
}

fn sword_map() -> &'static SwordMap {
    static M: OnceLock<SwordMap> = OnceLock::new();
    M.get_or_init(|| {
        let (w, h) = (2 * SWORD_HALF + 1, 2 * SWORD_HALF_Z + 1);
        // The header's lines have spaces; the map's rows (z from -SWORD_HALF_Z) have none.
        let rows: Vec<&[u8]> = include_str!("arena.txt").lines().filter(|l| !l.is_empty() && !l.contains(' ')).map(str::as_bytes).collect();
        assert!(rows.len() == h as usize && rows.iter().all(|r| r.len() == w as usize), "arena.txt: not {w} x {h}");
        let floor: Vec<bool> = (0..h).flat_map(|z| (0..w).map(move |x| (z, x))).map(|(z, x)| rows[z as usize][x as usize] == b'.').collect();
        let is_floor = |x: i32, z: i32| x.abs() <= SWORD_HALF && z.abs() <= SWORD_HALF_Z && floor[((z + SWORD_HALF_Z) * w + x + SWORD_HALF) as usize];
        let mut walls = Vec::new();
        for z in -SWORD_HALF_Z - 1..=SWORD_HALF_Z + 1 {
            for x in -SWORD_HALF - 1..=SWORD_HALF + 1 {
                if !is_floor(x, z) {
                    walls.push((x, z));
                }
            }
        }
        let box_d2 = |px: f64, pz: f64, (x, z): (i32, i32)| {
            let dx = px.clamp(x as f64, x as f64 + 1.0) - px;
            let dz = pz.clamp(z as f64, z as f64 + 1.0) - pz;
            dx * dx + dz * dz
        };
        let mut near = Vec::with_capacity((w * h) as usize);
        for z in -SWORD_HALF_Z..=SWORD_HALF_Z {
            for x in -SWORD_HALF..=SWORD_HALF {
                let (px, pz) = (x as f64 + 0.5, z as f64 + 0.5);
                near.push(*walls.iter().min_by(|a, b| box_d2(px, pz, **a).total_cmp(&box_d2(px, pz, **b))).unwrap());
            }
        }
        SwordMap { floor, near }
    })
}

/// Whether block column (x, z) is the sword arena's floor.
pub fn sword_floor(x: i32, z: i32) -> bool {
    x.abs() <= SWORD_HALF && z.abs() <= SWORD_HALF_Z && sword_map().floor[((z + SWORD_HALF_Z) * (2 * SWORD_HALF + 1) + x + SWORD_HALF) as usize]
}

/// From p to the nearest point of the wall cell nearest p's column's centre (x, z); (0, 0) off the floor.
pub fn sword_wall(p: V3) -> (f64, f64) {
    let (x, z) = (floor(p.x), floor(p.z));
    if !sword_floor(x, z) {
        return (0.0, 0.0);
    }
    let (wx, wz) = sword_map().near[((z + SWORD_HALF_Z) * (2 * SWORD_HALF + 1) + x + SWORD_HALF) as usize];
    (p.x.clamp(wx as f64, wx as f64 + 1.0) - p.x, p.z.clamp(wz as f64, wz as f64 + 1.0) - p.z)
}

/// The sword arena: the oval floor (everything below y = 1 solid), walled with bedrock up to the level's top and
/// solid outside it.
pub fn sword_arena() -> Level {
    let (hx, hz) = (SWORD_HALF + 1, SWORD_HALF_Z + 1);
    let mut level = Level::with([-hx, 1, -hz], [2 * hx + 1, HEIGHT - 1, 2 * hz + 1], |x, _, z| if sword_floor(x, z) { Block::Air } else { Block::Bedrock });
    level.floor = 1;
    level.walled = true;
    level
}

fn stand_y(level: &Level, x: i32, z: i32) -> i32 {
    let mut y = level.floor;
    while y < HEIGHT - 2 && (level.get(x, y, z).solid() || level.get(x, y + 1, z).solid()) {
        y += 1;
    }
    y
}

// ------------------------------------------------------------------ the ping

#[derive(Clone, Copy, Debug, Default)]
pub struct PingCfg {
    /// The round trip in ticks, drawn uniformly from this range for each side and fight (split at random between
    /// up and down).
    pub rtt: (u64, u64),
    /// Each tick: the chance that the delay each way moves to base + {-1, 0, 1}.
    pub jitter: f64,
    /// Each tick: the chance of a spike (2..=8 more ticks down and up for 5..=40 ticks).
    pub spike: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Ping {
    pub base: (u64, u64),
    pub now: (u64, u64),
    spike_left: u32,
    spike: u64,
}
impl Ping {
    pub fn new(rng: &mut Rng, cfg: &PingCfg) -> Ping {
        let rtt = rng.int(cfg.rtt.0 as i64, cfg.rtt.1 as i64) as u64;
        let up = rng.int(0, rtt as i64) as u64;
        let up = if rtt >= 2 { up.clamp(rtt / 4, rtt - rtt / 4) } else { up };
        let base = (up, rtt - up);
        Ping { base, now: base, spike_left: 0, spike: 0 }
    }
    pub fn tick(&mut self, rng: &mut Rng, cfg: &PingCfg) -> (u64, u64) {
        if self.spike_left > 0 {
            self.spike_left -= 1;
        } else if cfg.spike > 0.0 && rng.chance(cfg.spike) {
            self.spike_left = rng.int(5, 40) as u32;
            self.spike = rng.int(2, 8) as u64;
        } else {
            self.spike = 0;
        }
        if cfg.jitter > 0.0 && rng.chance(cfg.jitter) {
            let w = |b: u64, rng: &mut Rng| (b as i64 + rng.int(-1, 1)).max(0) as u64;
            self.now = (w(self.base.0, rng), w(self.base.1, rng));
        }
        let s = if self.spike_left > 0 { self.spike } else { 0 };
        (self.now.0 + s / 2, self.now.1 + s - s / 2)
    }
}

// ------------------------------------------------------------------ a fight

#[derive(Clone, Debug)]
pub struct FightCfg {
    pub max_ticks: u32,
    pub ping: [PingCfg; 2],
    /// Start distance between the two, in blocks (random starts).
    pub dist: (f64, f64),
    /// Chance of the fixed start (both at x = +-SWORD_SPAWN_X, facing each other); else both anywhere on the floor
    /// `dist` apart, facing each other or (half the time) any way.
    pub fixed_start: f64,
}
impl Default for FightCfg {
    fn default() -> Self {
        let p = PingCfg { rtt: (0, 6), jitter: 0.05, spike: 0.002 };
        FightCfg { max_ticks: 4800, ping: [p, p], dist: (4.0, 16.0), fixed_start: 0.0 }
    }
}

/// What a player shows the world (entity data every tracker gets), as the server has it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Public {
    pub pos: V3,
    pub health: f32,
    pub absorption: f32,
    /// The hurt animation's ticks left (10 on a hit), and the arm swing's ticks so far (-1: none).
    pub hurt_time: i32,
    pub swing_time: i32,
    pub sprinting: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    /// The other player died or left.
    Won(usize),
    Timeout,
}

#[derive(Clone)]
pub struct Fight {
    pub duel: Duel,
    pub rng: Rng,
    pub cfg: FightCfg,
    pub t: u32,
    pub ping: [Ping; 2],
    /// What each client has of the server, as its connection really delivers it (in order): one marker per server
    /// tick (the tick it reads at, the tick it was sent), and how many ticks old the newest state it has read is.
    pub down_q: [VecDeque<(u64, u64)>; 2],
    pub back: [u64; 2],
    /// The attack button each side holds: a click only presses it when it is up.
    held: [bool; 2],
    /// Each tick's public state of both, newest last (as long as the longest delay down needs).
    pub hist: VecDeque<[Public; 2]>,
    pub end: Option<End>,
}

impl Fight {
    pub fn new(seed: u64, cfg: &FightCfg) -> Fight {
        let mut rng = Rng(seed);
        let level = sword_arena();
        let (lo, span) = ([-SWORD_HALF as f64, -SWORD_HALF_Z as f64], [2.0 * SWORD_HALF as f64 + 1.0, 2.0 * SWORD_HALF_Z as f64 + 1.0]);
        let open = |p: V3| {
            let (x, z) = (floor(p.x), floor(p.z));
            let inside = p.x >= lo[0] && p.x < lo[0] + span[0] && p.z >= lo[1] && p.z < lo[1] + span[1];
            let y = stand_y(&level, x, z);
            inside && (-2..=1).contains(&y) && (y..y + 12).all(|k| !level.get(x, k, z).solid())
        };
        let real = rng.chance(cfg.fixed_start);
        let (a, b) = if real {
            let s = if rng.chance(0.5) { 1.0 } else { -1.0 };
            (V3::new(s * SWORD_SPAWN_X as f64, 0.0, 0.0), V3::new(-s * SWORD_SPAWN_X as f64, 0.0, 0.0))
        } else { loop {
            let a = V3::new(lo[0] + rng.f64() * span[0], 0.0, lo[1] + rng.f64() * span[1]);
            let d = cfg.dist.0 + rng.f64() * (cfg.dist.1 - cfg.dist.0);
            let ang = rng.f64() * std::f64::consts::TAU;
            let b = V3::new(a.x + d * ang.cos(), 0.0, a.z + d * ang.sin());
            if open(a) && open(b) {
                break (a, b);
            }
        } };
        let feet = |p: V3| {
            let (x, z) = (floor(p.x), floor(p.z));
            V3::new(x as f64 + 0.5, stand_y(&level, x, z) as f64, z as f64 + 0.5)
        };
        let pos = [feet(a), feet(b)];
        let yaw_to = |from: V3, to: V3| (-(to.x - from.x).atan2(to.z - from.z)).to_degrees() as f32;
        let mut yaws = [yaw_to(pos[0], pos[1]), yaw_to(pos[1], pos[0])];
        if !real && rng.chance(0.5) {
            yaws = [(rng.f64() * 360.0 - 180.0) as f32, (rng.f64() * 360.0 - 180.0) as f32];
        }
        let mut server = crate::sim::server::Server::new(level.clone());
        server.random = JavaRandom::new(rng.next_u64());
        let mut clients = Vec::new();
        for i in 0..2 {
            let id = i as i32 + 1;
            let (inv, armor, off) = sword_kit();
            let mut p = SPlayer::new(id, pos[i], yaws[i], 0.0);
            p.random = JavaRandom::new(rng.next_u64());
            p.inv = inv;
            p.armor = armor;
            p.offhand = off;
            p.body.on_ground = true;
            p.last_item_main = p.main_hand();
            p.last_equip_main = p.main_hand();
            for s in 0..41 {
                p.synced[s] = p.slot(s);
            }
            p.last_sent_health = p.health;
            p.last_sent_food = p.food.level;
            p.body.sneaking_speed = p.sneaking_speed();
            p.track.base = pos[i];
            p.track.yrot = pack_degrees(yaws[i]);
            p.track.xrot = pack_degrees(0.0);
            p.track.was_on_ground = true;
            server.players.push(p);
            server.out.push(Vec::new());
            let mut body = Body::new(pos[i], yaws[i], 0.0);
            body.on_ground = true;
            body.sneaking_speed = server.players[i].sneaking_speed();
            let mut c = Client::new(id, body, level.clone());
            c.inv = inv;
            c.armor = armor;
            c.offhand = off;
            c.last_item_main = c.main_hand();
            let o = 1 - i;
            let mut e = CEntity::new(o as i32 + 1, pos[o]);
            e.yaw = yaws[o];
            e.on_ground = true;
            c.entities.push(e);
            clients.push(Some(c));
        }
        server.next_id = 3;
        let mut duel = Duel::new(server, clients);
        let ping = [Ping::new(&mut rng, &cfg.ping[0]), Ping::new(&mut rng, &cfg.ping[1])];
        for i in 0..2 {
            duel.lag[i] = ping[i].now;
        }
        let mut f = Fight { duel, rng, cfg: cfg.clone(), t: 0, ping, down_q: [VecDeque::new(), VecDeque::new()], back: [0; 2], held: [false; 2], hist: VecDeque::new(), end: None };
        let p = f.public();
        f.hist.push_back(p);
        f
    }

    fn public(&self) -> [Public; 2] {
        let s = &self.duel.server;
        std::array::from_fn(|i| {
            let p = &s.players[i];
            Public {
                pos: p.body.pos,
                health: p.health,
                absorption: p.absorption,
                hurt_time: p.hurt_time,
                swing_time: if p.swinging { p.swing_time } else { -1 },
                sprinting: p.body.sprinting,
            }
        })
    }

    /// Player `j` as player `i`'s client would know it now: as the server had it `down` ticks ago.
    pub fn seen(&self, i: usize, j: usize) -> &Public {
        let back = self.back[i] as usize;
        let n = self.hist.len();
        &self.hist[n - 1 - back.min(n - 1)][j]
    }

    /// Mouse presses for a tick: `clicks` short clicks, then held down or not.
    fn presses(clicks: u32, hold: bool, held: &mut bool) -> u32 {
        let mut k = 0;
        for _ in 0..clicks {
            k += !*held as u32;
            *held = false;
        }
        if hold {
            k += !*held as u32;
            *held = true;
        } else if clicks == 0 {
            *held = false;
        }
        k
    }

    /// One tick with both sides' controls (`attacks` as clicks, `hold_attack` as held).
    pub fn step(&mut self, ctl: &mut [Controls; 2]) -> Option<End> {
        if self.end.is_some() {
            return self.end;
        }
        for i in 0..2 {
            ctl[i].attacks = Self::presses(ctl[i].attacks, ctl[i].hold_attack, &mut self.held[i]);
            let (up, down) = self.ping[i].tick(&mut self.rng, &self.cfg.ping[i]);
            self.duel.lag[i] = (up, down);
        }
        let now = self.duel.now;
        for i in 0..2 {
            // As Duel::sent and Duel::deliver: what the server sends this tick reads at now + down, in order.
            self.down_q[i].push_back((now + self.duel.lag[i].1, now));
            let mut newest = None;
            while self.down_q[i].front().is_some_and(|q| q.0 <= now) {
                newest = self.down_q[i].pop_front().map(|q| q.1);
            }
            self.back[i] = match newest {
                Some(sent) => now - sent,
                None => self.back[i] + 1,
            };
        }
        self.duel.tick(&ctl[..]);
        self.t += 1;
        let p = self.public();
        self.hist.push_back(p);
        let keep = 2 + (0..2).map(|i| self.ping[i].now.1.max(self.back[i]) + 8).max().unwrap() as usize;
        while self.hist.len() > keep.max(20) {
            self.hist.pop_front();
        }
        let s = &self.duel.server;
        let out = |p: &SPlayer| !s.level.has_column(floor(p.body.pos.x), floor(p.body.pos.z));
        let lost = [s.players[0].dead || out(&s.players[0]), s.players[1].dead || out(&s.players[1])];
        self.end = match lost {
            [true, true] => Some(End::Timeout),
            [true, false] => Some(End::Won(1)),
            [false, true] => Some(End::Won(0)),
            _ if self.t >= self.cfg.max_ticks => Some(End::Timeout),
            _ => None,
        };
        self.end
    }

    /// The opponent where side `i`'s client has it (its position and whether it crouches).
    pub fn opp_seen(&self, i: usize) -> Option<(V3, bool)> {
        let oid = self.duel.server.players[1 - i].id;
        self.client(i).entities.iter().find(|e| e.id == oid).map(|e| (e.pos, e.crouch))
    }

    pub fn view(&self, i: usize) -> View {
        let o = self.opp_seen(i);
        View { opp: o.map(|o| o.0), opp_crouch: o.is_some_and(|o| o.1) }
    }

    pub fn client(&self, i: usize) -> &Client {
        self.duel.clients[i].as_ref().unwrap()
    }
}

/// The opponent as a side's client has it.
pub struct View {
    pub opp: Option<V3>,
    pub opp_crouch: bool,
}

/// The yaw and pitch that look from `eye` at `p` (Entity.lookAt).
pub fn look_at(eye: V3, p: V3) -> (f32, f32) {
    let (dx, dy, dz) = (p.x - eye.x, p.y - eye.y, p.z - eye.z);
    let h = (dx * dx + dz * dz).sqrt();
    let pitch = wrap_degrees(-(dy.atan2(h).to_degrees()) as f32);
    let yaw = wrap_degrees((dz.atan2(dx).to_degrees() - 90.0) as f32);
    (yaw, pitch)
}

// ------------------------------------------------------------------ worker threads

/// Worker threads kept from step to step (spawning them anew costs a good part of a step), handing out the work a
/// few items at a time, as fights differ in cost.
pub struct Pool {
    shared: std::sync::Arc<PoolShared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}
#[derive(Clone, Copy)]
struct Task(*const (dyn Fn(usize) + Sync + 'static));
unsafe impl Send for Task {}
unsafe impl Sync for Task {}
struct PoolState {
    generation: u64,
    task: Option<Task>,
    n: usize,
    quit: bool,
}
struct PoolShared {
    state: std::sync::Mutex<PoolState>,
    wake: std::sync::Condvar,
    next: std::sync::atomic::AtomicUsize,
    /// Workers not yet through the current task.
    left: std::sync::atomic::AtomicUsize,
}
impl PoolShared {
    fn work(&self, task: Task, n: usize) {
        use std::sync::atomic::Ordering::Relaxed;
        const CHUNK: usize = 2;
        loop {
            let k = self.next.fetch_add(CHUNK, Relaxed);
            if k >= n {
                break;
            }
            for i in k..(k + CHUNK).min(n) {
                unsafe { (*task.0)(i) }
            }
        }
    }
}
impl Pool {
    pub fn new(threads: usize) -> Pool {
        let shared = std::sync::Arc::new(PoolShared {
            state: std::sync::Mutex::new(PoolState { generation: 0, task: None, n: 0, quit: false }),
            wake: std::sync::Condvar::new(),
            next: Default::default(),
            left: Default::default(),
        });
        let workers = (1..threads.max(1))
            .map(|_| {
                let sh = shared.clone();
                std::thread::spawn(move || {
                    let mut seen = 0;
                    loop {
                        let (task, n) = {
                            let mut st = sh.state.lock().unwrap();
                            while st.generation == seen && !st.quit {
                                st = sh.wake.wait(st).unwrap();
                            }
                            if st.quit {
                                return;
                            }
                            seen = st.generation;
                            (st.task.unwrap(), st.n)
                        };
                        sh.work(task, n);
                        sh.left.fetch_sub(1, std::sync::atomic::Ordering::Release);
                    }
                })
            })
            .collect();
        Pool { shared, workers }
    }
    /// f(0), .., f(n - 1) on all the threads; back when all are done.
    pub fn run(&self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        use std::sync::atomic::Ordering::{Acquire, Relaxed};
        if self.workers.is_empty() || n <= 1 {
            (0..n).for_each(f);
            return;
        }
        // The workers only use `f` until `left` is back to 0, which this waits for.
        let task = Task(unsafe { std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), *const (dyn Fn(usize) + Sync + 'static)>(f) });
        let sh = &self.shared;
        sh.next.store(0, Relaxed);
        sh.left.store(self.workers.len(), Relaxed);
        {
            let mut st = sh.state.lock().unwrap();
            st.generation += 1;
            st.task = Some(task);
            st.n = n;
        }
        sh.wake.notify_all();
        sh.work(task, n);
        let mut spins = 0u32;
        while sh.left.load(Acquire) > 0 {
            spins += 1;
            if spins < 1000 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        sh.state.lock().unwrap().task = None;
    }
}
impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().quit = true;
        self.shared.wake.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// A buffer the pool's tasks write their own disjoint parts of.
#[derive(Clone, Copy)]
pub(crate) struct Shared<T>(pub(crate) *mut T);
unsafe impl<T> Send for Shared<T> {}
unsafe impl<T> Sync for Shared<T> {}
impl<T> Shared<T> {
    /// Items off..off + n: no other task may touch them meanwhile.
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn part(&self, off: usize, n: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(off), n)
    }
}

