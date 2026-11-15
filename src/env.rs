// SPDX-License-Identifier: AGPL-3.0-only
// sword-sim by jo29_29. Licensed under the GNU AGPL v3 only with additional terms: see LICENSE and ADDITIONAL_TERMS.
//! Batches of sword fights for training (a C ABI for ctypes): the sword kit (diamond armor with Protection III and
//! Unbreaking III, an Unbreaking III diamond sword, nothing else; vanilla food) on the arena (fight::sword_arena: a
//! walled oval floor, 67 by 65), both sides at the fixed start (66 apart, facing each other) by
//! SwordConfig::fixed_start's chance, else anywhere on the floor.
//! The fight itself is fight::Fight: the bit-exact 1.21.11 server, both clients, and the ping.
//!
//! Per side and step:
//! - `obs` f32 [SOBS_DIM]: what its client knows (the opponent as its client has it, the rest as `Fight::seen`),
//!   in a frame turned to the opponent (u: the way to it, across; v: to its left), and its last actions;
//! - `priv` f32 [SPRIV_DIM]: the server's truth, for the critic only; priv[0] is the distance across /16;
//! - `phi`: the potential (health and absorption lead, /20, by the server's truth).
//!
//! Actions (SACT_DIM floats): fwd (0 back, 1 none, 2 forward), strafe (0 left, 1 none, 2 right), jump, sprint, sneak,
//! attack (a click), yaw off the way to the opponent's middle (-1..1: -180..180 degrees), pitch off it (-1..1:
//! -90..90). Rewards: the outcome only (+1 win, -1 loss, 0 timeout) on a fight's last step.
use crate::fight::*;
use crate::fight::{Pool, Shared};
use crate::sim::world::*;
use crate::sim::client::*;
use crate::sim::server::*;

pub const S_HIST: usize = 4;
pub const SACT_DIM: usize = 8;
pub const SSELF_DIM: usize = 30;
pub const SOPP_DIM: usize = 26;
pub const SOBS_DIM: usize = SSELF_DIM + SOPP_DIM + S_HIST * SACT_DIM;
pub const SPRIV_DIM: usize = 46;
/// outcome (0/1: that side won, 2: timeout), ticks, base rtt of 0 and 1, damage dealt by 0 and 1, hits landed by 0
/// and 1.
pub const SINFO_DIM: usize = 8;



/// The floor's half length, for positions in the observation.
const ARENA: f64 = SWORD_HALF as f64;

/// The nearest wall from p: how far (/8, up to 1), and its way in the frame (a unit vector; none when 8 or more away).
fn put_wall(w: &mut W, fr: &Frame, p: V3) {
    let (dx, dz) = sword_wall(p);
    let d = (dx * dx + dz * dz).sqrt();
    w.f((d / 8.0).min(1.0) as f32);
    let s = if d > 1e-9 && d < 8.0 { 1.0 / d } else { 0.0 };
    let (ux, uz) = fr.u;
    w.f(((dx * ux + dz * uz) * s) as f32);
    w.f(((dz * ux - dx * uz) * s) as f32);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SAct {
    pub fwd: i8,
    pub strafe: i8,
    pub jump: bool,
    pub sprint: bool,
    pub sneak: bool,
    pub attack: bool,
    pub yaw: f32,
    pub pitch: f32,
}

pub fn sdecode(a: &[f32]) -> SAct {
    let i = |k: usize| a[k].round() as i64;
    SAct {
        fwd: (i(0).clamp(0, 2) - 1) as i8,
        strafe: (i(1).clamp(0, 2) - 1) as i8,
        jump: a[2] > 0.5,
        sprint: a[3] > 0.5,
        sneak: a[4] > 0.5,
        attack: a[5] > 0.5,
        yaw: if a[6].is_finite() { a[6].clamp(-1.0, 1.0) } else { 0.0 },
        pitch: if a[7].is_finite() { a[7].clamp(-1.0, 1.0) } else { 0.0 },
    }
}

fn hist_row(a: &SAct, w: &mut W) {
    w.f(a.fwd as f32);
    w.f(a.strafe as f32);
    w.b(a.jump);
    w.b(a.sprint);
    w.b(a.sneak);
    w.b(a.attack);
    w.f(a.yaw);
    w.f(a.pitch);
}

/// The middle of a player standing at `p` (where the aim's zero looks).
fn middle(p: V3, crouch: bool) -> V3 {
    V3::new(p.x, p.y + if crouch { 0.75 } else { 0.9 }, p.z)
}

/// The keys and view for side `c`: the aim turns from the way to the opponent's middle (`opp`, where the client has
/// it), or from where it looks when it has none.
pub fn scontrols(a: &SAct, c: &Client, opp: Option<(V3, bool)>) -> Controls {
    scontrols_at(a, c.eye(), c.body.yaw, opp)
}

/// As `scontrols`, from the eye and yaw alone.
pub fn scontrols_at(a: &SAct, eye: V3, yaw: f32, opp: Option<(V3, bool)>) -> Controls {
    let (by, bp) = match opp {
        Some((o, crouch)) => look_at(eye, middle(o, crouch)),
        None => (yaw, 0.0),
    };
    Controls {
        keys: Keys { forward: a.fwd > 0, back: a.fwd < 0, left: a.strafe < 0, right: a.strafe > 0, jump: a.jump, shift: a.sneak, sprint: a.sprint },
        yaw: wrap_degrees(by + a.yaw * 180.0),
        pitch: (bp + a.pitch * 90.0).clamp(-90.0, 90.0),
        attacks: a.attack as u32,
        ..Default::default()
    }
}

struct W<'a> {
    o: &'a mut [f32],
    n: usize,
}
impl W<'_> {
    fn f(&mut self, v: f32) {
        self.o[self.n] = if v.is_finite() { v } else { 0.0 };
        self.n += 1;
    }
    fn b(&mut self, v: bool) {
        self.f(v as u8 as f32);
    }
}

fn sub(a: V3, b: V3) -> V3 {
    V3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

/// The frame turned to the opponent: u across towards it, v across to its left; a vector as (u, v, y).
#[derive(Clone, Copy)]
struct Frame {
    u: (f64, f64),
}
impl Frame {
    fn new(from: V3, to: V3, yaw: f32) -> Frame {
        let (dx, dz) = (to.x - from.x, to.z - from.z);
        let h = (dx * dx + dz * dz).sqrt();
        if h > 1e-6 {
            Frame { u: (dx / h, dz / h) }
        } else {
            // Where it looks (Entity.calculateViewVector across).
            let r = (yaw as f64).to_radians();
            Frame { u: (-r.sin(), r.cos()) }
        }
    }
    fn put(&self, w: &mut W, v: V3, s: f64) {
        let (ux, uz) = self.u;
        w.f(((v.x * ux + v.z * uz) * s) as f32);
        w.f(((v.z * ux - v.x * uz) * s) as f32);
        w.f((v.y * s) as f32);
    }
    /// A yaw's way (sin, cos) from the frame's u.
    fn turn(&self, w: &mut W, yaw: f32) {
        let r = (yaw as f64).to_radians();
        let (lx, lz) = (-r.sin(), r.cos());
        let (ux, uz) = self.u;
        w.f((lz * ux - lx * uz) as f32);
        w.f((lx * ux + lz * uz) as f32);
    }
}

/// What a side keeps track of between steps: its last actions and how long since things it saw happened.
#[derive(Clone, Copy, Debug, Default)]
pub struct Memory {
    pub hist: [SAct; S_HIST],
    /// Ticks since the opponent last started a swing, since its hurt animation last started, since mine did.
    since_swing: u32,
    since_op_hurt: u32,
    since_hurt: u32,
    last_op_swing: i32,
    last_op_hurt: i32,
    last_hurt: i32,
}

impl Memory {
    pub fn new() -> Memory {
        Memory { since_swing: 60, since_op_hurt: 60, since_hurt: 60, last_op_swing: -1, ..Default::default() }
    }
    fn tick(&mut self, f: &Fight, i: usize) {
        let (op, me) = (f.seen(i, 1 - i), f.seen(i, i));
        self.saw(op.swing_time, op.hurt_time, me.hurt_time);
    }
    /// A tick's news: the opponent's arm swing (ticks so far, -1 none), its hurt animation and mine.
    pub fn saw(&mut self, op_swing: i32, op_hurt: i32, my_hurt: i32) {
        let up = |n: &mut u32| *n = (*n + 1).min(60);
        up(&mut self.since_swing);
        up(&mut self.since_op_hurt);
        up(&mut self.since_hurt);
        if op_swing >= 0 && (self.last_op_swing < 0 || op_swing < self.last_op_swing) {
            self.since_swing = op_swing as u32;
        }
        if op_hurt > self.last_op_hurt {
            self.since_op_hurt = 0;
        }
        if my_hurt > self.last_hurt {
            self.since_hurt = 0;
        }
        self.last_op_swing = op_swing;
        self.last_op_hurt = op_hurt;
        self.last_hurt = my_hurt;
    }
}

/// What a client knows, all an observation is made from (filled in from the sim's `Fight`).
#[derive(Clone, Copy, Debug)]
pub struct SKnown {
    pub health: f32,
    pub absorption: f32,
    pub food: i32,
    pub saturation: f32,
    pub pos: V3,
    pub eye: V3,
    pub vel: V3,
    pub yaw: f32,
    pub pitch: f32,
    pub on_ground: bool,
    pub sprinting: bool,
    pub crouch_pose: bool,
    pub horizontal_collision: bool,
    pub fall_distance: f64,
    pub attack_strength_ticker: i32,
    pub miss_time: i32,
    pub hurt_time: i32,
    /// The ping's delay up and down, in ticks.
    pub ping: (u64, u64),
    /// The fight's share of time left, 1 to 0.
    pub time_left: f32,
    pub op_pos: V3,
    pub op_crouch: bool,
    /// The opponent's entity on this client, if it has one.
    pub op_entity: Option<SEntity>,
    pub op_health: f32,
    pub op_absorption: f32,
    pub op_sprinting: bool,
    pub op_hurt_time: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct SEntity {
    pub pos: V3,
    /// Its move over the last tick.
    pub delta: V3,
    pub on_ground: bool,
    pub crouch: bool,
    pub yaw: f32,
    pub pitch: f32,
}

/// Side `i`'s observation of fight `f`.
pub fn sobserve(f: &Fight, i: usize, m: &Memory, obs: &mut [f32]) {
    let c = f.client(i);
    let view = f.view(i);
    let me_seen = f.seen(i, i);
    let op = f.seen(i, 1 - i);
    let oid = f.duel.server.players[1 - i].id;
    let oe = c.entities.iter().find(|e| e.id == oid);
    let k = SKnown {
        health: c.health,
        absorption: c.absorption,
        food: c.food,
        saturation: c.saturation,
        pos: c.body.pos,
        eye: c.eye(),
        vel: c.body.vel,
        yaw: c.body.yaw,
        pitch: c.body.pitch,
        on_ground: c.body.on_ground,
        sprinting: c.body.sprinting,
        crouch_pose: c.body.crouch_pose,
        horizontal_collision: c.body.horizontal_collision,
        fall_distance: c.body.fall_distance,
        attack_strength_ticker: c.attack_strength_ticker,
        miss_time: c.miss_time,
        hurt_time: me_seen.hurt_time,
        ping: f.ping[i].base,
        time_left: 1.0 - f.t as f32 / f.cfg.max_ticks as f32,
        op_pos: view.opp.unwrap_or(op.pos),
        op_crouch: view.opp_crouch,
        op_entity: oe.map(|e| SEntity { pos: e.pos, delta: e.prev_pos.map_or(V3::ZERO, |p| sub(e.pos, p)), on_ground: e.on_ground, crouch: e.crouch, yaw: e.yaw, pitch: e.pitch }),
        op_health: op.health,
        op_absorption: op.absorption,
        op_sprinting: op.sprinting,
        op_hurt_time: op.hurt_time,
    };
    sobserve_known(&k, m, obs);
}

/// The observation from what a client knows.
pub fn sobserve_known(k: &SKnown, m: &Memory, obs: &mut [f32]) {
    let my_pos = k.pos;
    let op_pos = k.op_pos;
    let fr = Frame::new(my_pos, op_pos, k.yaw);
    let eye = k.eye;
    let (aim_yaw, aim_pitch) = look_at(eye, middle(op_pos, k.op_crouch));
    let mut w = W { o: obs, n: 0 };

    // Me (SSELF_DIM).
    w.f(k.health / 20.0);
    w.f(k.absorption / 20.0);
    w.f(k.food as f32 / 20.0);
    w.f(k.saturation / 20.0);
    fr.put(&mut w, k.vel, 2.0);
    w.b(k.on_ground);
    w.b(k.sprinting);
    w.b(k.crouch_pose);
    w.b(k.horizontal_collision);
    w.f((k.fall_distance.min(4.0) / 4.0) as f32);
    w.f(k.attack_strength_ticker.min(13) as f32 / 13.0);
    w.b(k.attack_strength_ticker >= 12);
    w.b(k.miss_time > 0);
    w.f(k.hurt_time as f32 / 10.0);
    w.f(m.since_hurt as f32 / 60.0);
    // Where I look: off the aim at the opponent's middle, and the pitch itself.
    let dy = (wrap_degrees(k.yaw - aim_yaw) as f64).to_radians();
    w.f(dy.sin() as f32);
    w.f(dy.cos() as f32);
    w.f((k.pitch - aim_pitch) / 90.0);
    w.f(k.pitch / 90.0);
    // The centre of the floor from me, and the nearest wall.
    fr.put(&mut w, V3::new(0.5 - my_pos.x, 0.0, 0.5 - my_pos.z), 1.0 / ARENA);
    put_wall(&mut w, &fr, my_pos);
    w.f(k.ping.0 as f32 / 10.0);
    w.f(k.ping.1 as f32 / 10.0);
    w.f(k.time_left);
    assert_eq!(w.n, SSELF_DIM);

    // The opponent (SOPP_DIM).
    let to = sub(op_pos, my_pos);
    let across = (to.x * to.x + to.z * to.z).sqrt();
    let reach = len3(sub(middle(op_pos, k.op_crouch), eye));
    w.b(k.op_entity.is_some());
    w.f((across / 8.0).min(4.0) as f32);
    w.f(((1.0 + across).ln() / 3.0) as f32);
    w.f((to.y / 4.0) as f32);
    w.f((reach / 3.0).min(4.0) as f32);
    w.b(reach <= 3.0);
    match k.op_entity {
        Some(e) => {
            fr.put(&mut w, e.delta, 2.0);
            w.b(e.on_ground);
            w.b(e.crouch);
            // Where it looks, off the way back to me.
            let back = look_at(V3::new(e.pos.x, e.pos.y + 1.62, e.pos.z), eye).0;
            let r = (wrap_degrees(e.yaw - back) as f64).to_radians();
            w.f(r.sin() as f32);
            w.f(r.cos() as f32);
            w.f(e.pitch / 90.0);
        }
        None => {
            for _ in 0..8 {
                w.f(0.0);
            }
        }
    }
    w.f(k.op_health / 20.0);
    w.f(k.op_absorption / 20.0);
    w.b(k.op_sprinting);
    w.f(k.op_hurt_time as f32 / 10.0);
    w.f(m.since_op_hurt as f32 / 60.0);
    w.f(m.since_swing as f32 / 60.0);
    fr.put(&mut w, V3::new(0.5 - op_pos.x, 0.0, 0.5 - op_pos.z), 1.0 / ARENA);
    put_wall(&mut w, &fr, op_pos);
    assert_eq!(w.n, SSELF_DIM + SOPP_DIM);

    for a in &m.hist {
        hist_row(a, &mut w);
    }
    assert_eq!(w.n, SOBS_DIM);
}

fn len3(v: V3) -> f64 {
    (v.x * v.x + v.y * v.y + v.z * v.z).sqrt()
}

/// The critic's extra: the server's truth. priv[0]: the distance across /16.
pub fn sprivileged(f: &Fight, i: usize, out: &mut [f32]) {
    let s = &f.duel.server;
    let (a, b) = (&s.players[i], &s.players[1 - i]);
    let fr = Frame::new(a.body.pos, b.body.pos, a.body.yaw);
    let to = sub(b.body.pos, a.body.pos);
    let mut w = W { o: out, n: 0 };
    w.f(((to.x * to.x + to.z * to.z).sqrt() / 16.0) as f32);
    w.f((to.y / 4.0) as f32);
    fr.put(&mut w, a.body.vel, 2.0);
    fr.put(&mut w, b.body.vel, 2.0);
    for p in [a, b] {
        w.f(p.health / 20.0);
        w.f(p.absorption / 20.0);
        w.f(p.food.level as f32 / 20.0);
        w.f(p.food.saturation / 20.0);
        w.f(p.attack_strength_ticker.min(13) as f32 / 13.0);
        w.f(p.invulnerable_time as f32 / 20.0);
        w.f(p.hurt_time as f32 / 10.0);
        w.f((p.body.fall_distance.min(4.0) / 4.0) as f32);
        w.b(p.body.sprinting);
        w.b(p.body.on_ground);
        fr.turn(&mut w, p.body.yaw);
        w.f(p.body.pitch / 90.0);
    }
    for k in [i, 1 - i] {
        // The delay each way.
        w.f(f.ping[k].now.0 as f32 / 10.0);
        w.f(f.ping[k].now.1 as f32 / 10.0);
    }
    w.f(1.0 - f.t as f32 / f.cfg.max_ticks as f32);
    w.f(spotential(f, i));
    fr.put(&mut w, V3::new(-a.body.pos.x, 0.0, -a.body.pos.z), 1.0 / ARENA);
    fr.put(&mut w, V3::new(-b.body.pos.x, 0.0, -b.body.pos.z), 1.0 / ARENA);
    assert_eq!(w.n, SPRIV_DIM);
}

/// The potential: my health and absorption less the opponent's, /20 (the server's truth).
pub fn spotential(f: &Fight, i: usize) -> f32 {
    let s = &f.duel.server;
    let hp = |k: usize| if s.players[k].dead { 0.0 } else { s.players[k].health + s.players[k].absorption };
    (hp(i) - hp(1 - i)) / 20.0
}

// ------------------------------------------------------------------ a batch

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SwordConfig {
    pub max_ticks: u32,
    pub rtt_lo: u32,
    pub rtt_hi: u32,
    pub jitter: f32,
    pub spike: f32,
    pub dist_lo: f32,
    pub dist_hi: f32,
    /// The chance of the fixed start (else both anywhere on the floor, dist_lo..dist_hi apart).
    pub fixed_start: f32,
}
impl Default for SwordConfig {
    fn default() -> Self {
        SwordConfig { max_ticks: 3600, rtt_lo: 0, rtt_hi: 6, jitter: 0.05, spike: 0.002, dist_lo: 2.5, dist_hi: 16.0, fixed_start: 0.5 }
    }
}

impl SwordConfig {
    fn fight(&self) -> FightCfg {
        let p = PingCfg { rtt: (self.rtt_lo as u64, self.rtt_hi.max(self.rtt_lo) as u64), jitter: self.jitter as f64, spike: self.spike as f64 };
        FightCfg { max_ticks: self.max_ticks, ping: [p, p], dist: (self.dist_lo as f64, self.dist_hi.max(self.dist_lo) as f64), fixed_start: self.fixed_start as f64, ..Default::default() }
    }
}

pub struct SSlot {
    pub fight: Fight,
    mem: [Memory; 2],
    hits: [u32; 2],
    /// Each player's hurt time after the last tick (a rise is a hit).
    hurt: [i32; 2],
    seed: u64,
}

impl SSlot {
    fn new(seed: u64, cfg: &SwordConfig) -> SSlot {
        let fight = Fight::new(seed, &cfg.fight());
        let mut s = SSlot { fight, mem: [Memory::new(); 2], hits: [0; 2], hurt: [0; 2], seed };
        for i in 0..2 {
            s.mem[i].tick(&s.fight, i);
        }
        s
    }
    fn write(&self, i: usize, obs: &mut [f32], prv: &mut [f32], phi: &mut f32) {
        sobserve(&self.fight, i, &self.mem[i], obs);
        sprivileged(&self.fight, i, prv);
        *phi = spotential(&self.fight, i);
    }
    fn step(&mut self, cfg: &SwordConfig, actions: &[f32], reward: &mut [f32], done: &mut u8, info: &mut [f32]) {
        let mut acts = [SAct::default(); 2];
        for i in 0..2 {
            acts[i] = sdecode(&actions[i * SACT_DIM..][..SACT_DIM]);
        }
        let mut ctl: [Controls; 2] = std::array::from_fn(|i| {
            scontrols(&acts[i], self.fight.client(i), self.fight.opp_seen(i))
        });
        for i in 0..2 {
            let h = &mut self.mem[i].hist;
            h.rotate_right(1);
            h[0] = acts[i];
        }
        let end = self.fight.step(&mut ctl);
        // A hit landed this tick (the only damage there is): the other side's hurt animation started.
        for k in 0..2 {
            let h = self.fight.duel.server.players[1 - k].hurt_time;
            if h > self.hurt[1 - k] {
                self.hits[k] += 1;
            }
            self.hurt[1 - k] = h;
        }
        reward[0] = 0.0;
        reward[1] = 0.0;
        *done = 0;
        if let Some(e) = end {
            let f = &self.fight;
            let s = &f.duel.server;
            let outcome = match e {
                End::Won(w) => {
                    reward[w] = 1.0;
                    reward[1 - w] = -1.0;
                    w as f32
                }
                End::Timeout => 2.0,
            };
            let row = [
                outcome,
                f.t as f32,
                (f.ping[0].base.0 + f.ping[0].base.1) as f32,
                (f.ping[1].base.0 + f.ping[1].base.1) as f32,
                s.stats[1].by_other,
                s.stats[0].by_other,
                self.hits[0] as f32,
                self.hits[1] as f32,
            ];
            info.copy_from_slice(&row);
            *done = 1;
            let seed = Rng(self.seed ^ 0x5DEE_CE66_D1CE_4E5B).next_u64();
            *self = SSlot::new(seed, cfg);
            return;
        }
        for i in 0..2 {
            self.mem[i].tick(&self.fight, i);
        }
    }
}

pub struct Sword {
    pub slots: Vec<SSlot>,
    pub cfg: SwordConfig,
    pool: Pool,
}

impl Sword {
    pub fn new(n: usize, seed: u64, threads: usize) -> Sword {
        let cfg = SwordConfig::default();
        let slots = (0..n).map(|k| SSlot::new(Rng(seed.wrapping_mul(0x1000_0000_01B3) ^ k as u64).next_u64(), &cfg)).collect();
        Sword { slots, cfg, pool: Pool::new(threads) }
    }
    pub fn reset(&mut self) {
        let cfg = self.cfg;
        let sl = Shared(self.slots.as_mut_ptr());
        self.pool.run(self.slots.len(), &|k| {
            let s = unsafe { &mut sl.part(k, 1)[0] };
            *s = SSlot::new(s.seed.wrapping_add(0x9E37_79B9), &cfg);
        });
    }
    pub fn observe(&mut self, obs: &mut [f32], prv: &mut [f32], phi: &mut [f32]) {
        for (k, sl) in self.slots.iter().enumerate() {
            for i in 0..2 {
                let j = 2 * k + i;
                sl.write(i, &mut obs[j * SOBS_DIM..][..SOBS_DIM], &mut prv[j * SPRIV_DIM..][..SPRIV_DIM], &mut phi[j]);
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn step(&mut self, actions: &[f32], obs: &mut [f32], prv: &mut [f32], phi: &mut [f32], reward: &mut [f32], done: &mut [u8], info: &mut [f32]) {
        let cfg = self.cfg;
        let n = self.slots.len();
        assert!(actions.len() >= n * 2 * SACT_DIM && obs.len() >= n * 2 * SOBS_DIM && prv.len() >= n * 2 * SPRIV_DIM);
        assert!(phi.len() >= n * 2 && reward.len() >= n * 2 && done.len() >= n && info.len() >= n * SINFO_DIM);
        let sl = Shared(self.slots.as_mut_ptr());
        let (ob, pr, ph, re, dn, inf) = (Shared(obs.as_mut_ptr()), Shared(prv.as_mut_ptr()), Shared(phi.as_mut_ptr()), Shared(reward.as_mut_ptr()), Shared(done.as_mut_ptr()), Shared(info.as_mut_ptr()));
        self.pool.run(n, &|k| unsafe {
            let s = &mut sl.part(k, 1)[0];
            s.step(&cfg, &actions[k * 2 * SACT_DIM..][..2 * SACT_DIM], re.part(2 * k, 2), &mut dn.part(k, 1)[0], inf.part(k * SINFO_DIM, SINFO_DIM));
            for i in 0..2 {
                let j = 2 * k + i;
                s.write(i, ob.part(j * SOBS_DIM, SOBS_DIM), pr.part(j * SPRIV_DIM, SPRIV_DIM), &mut ph.part(j, 1)[0]);
            }
        });
    }
}

// ------------------------------------------------------------------ C ABI

#[no_mangle]
pub extern "C" fn sword_dims(out: *mut usize) {
    let dims = [SOBS_DIM, SPRIV_DIM, SACT_DIM, SINFO_DIM, SSELF_DIM, SOPP_DIM, S_HIST];
    unsafe { std::ptr::copy_nonoverlapping(dims.as_ptr(), out, dims.len()) }
}
#[no_mangle]
pub extern "C" fn sword_new(n: usize, seed: u64, threads: usize) -> *mut Sword {
    Box::into_raw(Box::new(Sword::new(n, seed, threads)))
}
#[no_mangle]
pub unsafe extern "C" fn sword_free(s: *mut Sword) {
    drop(Box::from_raw(s));
}
#[no_mangle]
pub unsafe extern "C" fn sword_configure(s: *mut Sword, cfg: SwordConfig) {
    (*s).cfg = cfg;
}
#[no_mangle]
pub unsafe extern "C" fn sword_reset(s: *mut Sword) {
    (*s).reset();
}
#[no_mangle]
pub unsafe extern "C" fn sword_observe(s: *mut Sword, obs: *mut f32, prv: *mut f32, phi: *mut f32) {
    use std::slice::from_raw_parts_mut as m;
    let s = &mut *s;
    let n = s.slots.len();
    s.observe(m(obs, n * 2 * SOBS_DIM), m(prv, n * 2 * SPRIV_DIM), m(phi, n * 2));
}
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn sword_step(s: *mut Sword, actions: *const f32, obs: *mut f32, prv: *mut f32, phi: *mut f32, reward: *mut f32, done: *mut u8, info: *mut f32) {
    use std::slice::from_raw_parts_mut as m;
    let s = &mut *s;
    let n = s.slots.len();
    s.step(std::slice::from_raw_parts(actions, n * 2 * SACT_DIM), m(obs, n * 2 * SOBS_DIM), m(prv, n * 2 * SPRIV_DIM), m(phi, n * 2), m(reward, n * 2), m(done, n), m(info, n * SINFO_DIM));
}
/// Slot `k`'s state for a viewer: per side x, y, z, yaw, pitch, health, absorption, hurt time; then the tick.
#[no_mangle]
pub unsafe extern "C" fn sword_state(s: *mut Sword, k: usize, out: *mut f64) {
    let s = &*s;
    let f = &s.slots[k].fight;
    let o = std::slice::from_raw_parts_mut(out, 17);
    for i in 0..2 {
        let p = &f.duel.server.players[i];
        o[8 * i..][..8].copy_from_slice(&[p.body.pos.x, p.body.pos.y, p.body.pos.z, p.body.yaw as f64, p.body.pitch as f64, p.health as f64, p.absorption as f64, p.hurt_time as f64]);
    }
    o[16] = f.t as f64;
}
