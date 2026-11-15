// SPDX-License-Identifier: AGPL-3.0-only
// sword-sim by jo29_29. Licensed under the GNU AGPL v3 only with additional terms: see LICENSE and ADDITIONAL_TERMS.
//! A 1.21.11 sword duel tick by tick: the clients (LocalPlayer, Minecraft.tick's clicks and picks, what they
//! predict and what the server's packets change) and the order of one tick, as the integrated server runs it:
//!
//! 1. the server handles what each client sent in its last tick (PacketProcessor, at the start of the tick);
//! 2. the server ticks (sendChanges, the entities, each connection's doTick);
//! 3. each client ticks (its packets go to the next server tick);
//! 4. each client handles the server's packets.
//!
use std::collections::VecDeque;

use crate::sim::world::*;
use crate::sim::server::*;
use crate::sim::server::Server;

/// A click or key this tick, on top of the held keys.
#[derive(Clone, Debug, Default)]
pub struct Controls {
    pub keys: Keys,
    pub yaw: f32,
    pub pitch: f32,
    /// Attack clicks this tick, and whether the button stays held after them.
    pub attacks: u32,
    pub hold_attack: bool,
}

/// What the crosshair is on (Minecraft.hitResult).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hit {
    Miss,
    Block(BlockHit),
    Entity(i32, V3),
}

/// An entity as a client sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CEntity {
    pub id: i32,
    pub pos: V3,
    pub crouch: bool,
    pub yaw: f32,
    pub pitch: f32,
    pub on_ground: bool,
    /// Entity.positionCodec's base: what move packets are relative to.
    pub base: V3,
    /// The InterpolationHandler (other players: 3 steps): steps left, the target, last tick's position and angles.
    pub steps: i32,
    pub target: V3,
    pub tyaw: f32,
    pub tpitch: f32,
    pub prev_pos: Option<V3>,
    /// (xRot, yRot), as the handler's Vec2.
    pub prev_rot: Option<(f32, f32)>,
}
impl CEntity {
    pub fn new(id: i32, pos: V3) -> Self {
        CEntity {
            id,
            pos,
            crouch: false,
            yaw: 0.0,
            pitch: 0.0,
            on_ground: false,
            base: pos,
            steps: 0,
            target: V3::ZERO,
            tyaw: 0.0,
            tpitch: 0.0,
            prev_pos: None,
            prev_rot: None,
        }
    }
    fn i_pos(&self) -> V3 {
        if self.steps > 0 { self.target } else { self.pos }
    }
    fn i_yaw(&self) -> f32 {
        if self.steps > 0 { self.tyaw } else { self.yaw }
    }
    fn i_pitch(&self) -> f32 {
        if self.steps > 0 { self.tpitch } else { self.pitch }
    }
    /// VecDeltaCodec.decode.
    fn decode(&self, l: [i64; 3]) -> V3 {
        if l == [0, 0, 0] {
            return self.base;
        }
        let one = |b: f64, d: i64| if d == 0 { b } else { (delta_encode(b) + d) as f64 / 4096.0 };
        V3::new(one(self.base.x, l[0]), one(self.base.y, l[1]), one(self.base.z, l[2]))
    }
    /// InterpolationHandler.interpolateTo: a new target unless it is the one already being approached.
    fn interpolate_to(&mut self, v: V3, f: f32, g: f32) {
        let same = |a: V3, b: V3| a.x.to_bits() == b.x.to_bits() && a.y.to_bits() == b.y.to_bits() && a.z.to_bits() == b.z.to_bits();
        if !(self.steps > 0) || self.i_yaw().to_bits() != f.to_bits() || self.i_pitch().to_bits() != g.to_bits() || !same(self.i_pos(), v) {
            self.steps = 3;
            self.target = v;
            self.tyaw = f;
            self.tpitch = g;
            self.prev_pos = Some(self.pos);
            self.prev_rot = Some((self.pitch, self.yaw));
        }
    }
    /// Entity.setRot (setXRot clamps).
    fn set_rot(&mut self, f: f32, g: f32) {
        self.yaw = f % 360.0;
        self.pitch = ((g % 360.0) % 360.0).clamp(-90.0, 90.0);
    }
    /// InterpolationHandler.interpolate: a third (half, all) of the way to the target.
    fn interpolate(&mut self, level: &Level) {
        if self.steps <= 0 {
            self.steps = 0;
            self.prev_pos = None;
            self.prev_rot = None;
            return;
        }
        let d = 1.0 / self.steps as f64;
        if let Some(pp) = self.prev_pos {
            let v = self.pos.sub(pp);
            let moved = self.target.add(v);
            let mut c = self.clone();
            c.pos = moved;
            if !level_collides(level, &c.bb()) {
                self.target = moved;
            }
        }
        if let Some((px, py)) = self.prev_rot {
            let f = self.yaw - py;
            let g = self.pitch - px;
            self.tyaw += f;
            self.tpitch += g;
        }
        let lerp = |d: f64, a: f64, b: f64| a + d * (b - a);
        let wrap = |x: f64| {
            let mut e = x % 360.0;
            if e >= 180.0 {
                e -= 360.0;
            }
            if e < -180.0 {
                e += 360.0;
            }
            e
        };
        let np = V3::new(lerp(d, self.pos.x, self.target.x), lerp(d, self.pos.y, self.target.y), lerp(d, self.pos.z, self.target.z));
        let j = (self.yaw as f64 + d * wrap(self.tyaw as f64 - self.yaw as f64)) as f32;
        let k = lerp(d, self.pitch as f64, self.tpitch as f64) as f32;
        self.pos = np;
        self.set_rot(j, k);
        self.steps -= 1;
        self.prev_pos = Some(np);
        self.prev_rot = Some((self.pitch, self.yaw));
    }
}

fn intersects_bb(a: &Bb, b: &Bb) -> bool {
    a.min[0] < b.max[0] && a.max[0] > b.min[0] && a.min[1] < b.max[1] && a.max[1] > b.min[1] && a.min[2] < b.max[2] && a.max[2] > b.min[2]
}

/// Level.noCollision for a box, blocks only (nothing else in the arena collides).
fn level_collides(level: &Level, b: &Bb) -> bool {
    let (x0, x1) = ((b.min[0] - 1.0e-7).floor() as i32 - 1, (b.max[0] + 1.0e-7).floor() as i32 + 1);
    let (y0, y1) = ((b.min[1] - 1.0e-7).floor() as i32 - 1, (b.max[1] + 1.0e-7).floor() as i32 + 1);
    let (z0, z1) = ((b.min[2] - 1.0e-7).floor() as i32 - 1, (b.max[2] + 1.0e-7).floor() as i32 + 1);
    for x in x0..=x1 {
        for y in y0..=y1 {
            for z in z0..=z1 {
                if level.get(x, y, z).solid() {
                    let c = Bb { min: [x as f64, y as f64, z as f64], max: [x as f64 + 1.0, y as f64 + 1.0, z as f64 + 1.0] };
                    if b.min[0] < c.max[0] && b.max[0] > c.min[0] && b.min[1] < c.max[1] && b.max[1] > c.min[1] && b.min[2] < c.max[2] && b.max[2] > c.min[2] {
                        return true;
                    }
                }
            }
        }
    }
    false
}

impl CEntity {
    pub fn bb(&self) -> Bb {
        let f = (0.6f32 / 2.0f32) as f64;
        let h = if self.crouch { 1.5f32 } else { 1.8f32 } as f64;
        Bb { min: [self.pos.x - f, self.pos.y, self.pos.z - f], max: [self.pos.x + f, self.pos.y + h, self.pos.z + f] }
    }
}



/// A client: its player, what it believes about the world, and its click timers.
#[derive(Clone, Debug)]
pub struct Client {
    pub id: i32,
    /// Entity reach of its crosshair (vanilla 3).
    pub reach: f64,
    /// How many motion packets for its own player it has had (knockback).
    pub motions: u32,
    pub body: Body,
    pub old_pos: V3,
    pub level: Level,
    pub entities: Vec<CEntity>,
    pub inv: [Stack; 36],
    pub armor: [Stack; 4],
    pub offhand: Stack,
    pub selected: usize,
    pub carried_index: usize,
    pub attack_strength_ticker: i32,
    pub last_item_main: Stack,
    pub miss_time: i32,
    pub health: f32,
    pub absorption: f32,
    pub food: i32,
    pub saturation: f32,
    pub hit: Hit,
    // LocalPlayer.sendPosition's memory.
    pub last_pos: V3,
    pub y_rot_last: f32,
    pub x_rot_last: f32,
    pub position_reminder: i32,
    pub last_on_ground: bool,
    pub last_hcol: bool,
    pub last_sent_input: Keys,
    pub km_attack: bool,
    /// MultiPlayerGameMode's block breaking (bedrock: it never breaks): whether it is at it, the block and the item
    /// it started with.
    pub is_destroying: bool,
    pub destroy_pos: [i32; 3],
    pub destroying_item: Stack,
    pub out: Vec<C2S>,
}


impl Client {

    pub fn new(id: i32, body: Body, level: Level) -> Self {
        let p = body.pos;
        let (yaw, pitch, ground) = (body.yaw, body.pitch, body.on_ground);
        Client {
            id,
            reach: 3.0,
            motions: 0,
            body,
            old_pos: p,
            level,
            entities: Vec::new(),
            inv: [Stack::EMPTY; 36],
            armor: [Stack::EMPTY; 4],
            offhand: Stack::EMPTY,
            selected: 0,
            carried_index: 0,
            attack_strength_ticker: 0,
            last_item_main: Stack::EMPTY,
            miss_time: 0,
            health: 20.0,
            absorption: 0.0,
            food: 20,
            saturation: 5.0,
            hit: Hit::Miss,
            last_pos: p,
            y_rot_last: yaw,
            x_rot_last: pitch,
            position_reminder: 0,
            last_on_ground: ground,
            last_hcol: false,
            last_sent_input: Keys::default(),
            km_attack: false,
            is_destroying: false,
            destroy_pos: [-1, -1, -1],
            destroying_item: Stack::EMPTY,
            out: Vec::new(),
        }
    }
    pub fn main_hand(&self) -> Stack {
        self.inv[self.selected]
    }
    pub fn eye(&self) -> V3 {
        // getEyePosition(1.0): lerp from the last tick's position.
        let p = self.body.pos;
        let o = self.old_pos;
        V3::new(o.x + 1.0 * (p.x - o.x), o.y + 1.0 * (p.y - o.y) + self.body.eye_height() as f64, o.z + 1.0 * (p.z - o.z))
    }

    /// GameRenderer.pick -> LocalPlayer.raycastHitResult (block reach 4.5, entity reach 3).
    pub fn pick(&self) -> Hit {
        self.pick_dir(self.body.yaw, self.body.pitch)
    }

    /// What `pick` would find looking at (yaw, pitch) from here.
    pub fn pick_dir(&self, yaw: f32, pitch: f32) -> Hit {
        let (d, e) = (4.5f64, self.reach);
        let mut g = jmax(d, e);
        let mut h = g * g;
        let eye = self.eye();
        let view = view_vector(pitch, yaw);
        let end = eye.add(V3::new(view.x * g, view.y * g, view.z * g));
        let block = clip(&self.level, eye, end, true);
        let loc = match &block {
            Ok(b) => b.loc,
            Err((l, _)) => *l,
        };
        let i = loc.distance_to_sqr(eye);
        if block.is_ok() {
            h = i;
            g = i.sqrt();
        }
        let end2 = eye.add(V3::new(view.x * g, view.y * g, view.z * g));
        let area = {
            let b = self.body.bounding_box().expand_towards(view.scale(g));
            Bb { min: [b.min[0] - 1.0, b.min[1] - 1.0, b.min[2] - 1.0], max: [b.max[0] + 1.0, b.max[1] + 1.0, b.max[2] + 1.0] }
        };
        // ProjectileUtil.getEntityHitResult.
        let mut best: Option<(i32, V3)> = None;
        let mut ee = h;
        for ent in &self.entities {
            let bb = ent.bb();
            if !(bb.min[0] < area.max[0] && bb.max[0] > area.min[0] && bb.min[1] < area.max[1] && bb.max[1] > area.min[1] && bb.min[2] < area.max[2] && bb.max[2] > area.min[2]) {
                continue;
            }
            let hit = aabb_clip(&bb, eye, end2);
            let inside = eye.x >= bb.min[0] && eye.x < bb.max[0] && eye.y >= bb.min[1] && eye.y < bb.max[1] && eye.z >= bb.min[2] && eye.z < bb.max[2];
            if inside {
                if ee >= 0.0 {
                    best = Some((ent.id, hit.unwrap_or(eye)));
                    ee = 0.0;
                }
            } else if let Some(v) = hit {
                let f = eye.distance_to_sqr(v);
                if f < ee || ee == 0.0 {
                    best = Some((ent.id, v));
                    ee = f;
                }
            }
        }
        let filter = |loc: V3, r: f64| loc.distance_to_sqr(eye) < r * r;
        if let Some((id, v)) = best {
            if v.distance_to_sqr(eye) < i {
                return if filter(v, e) { Hit::Entity(id, v) } else { Hit::Miss };
            }
        }
        match block {
            Ok(b) if filter(b.loc, d) => Hit::Block(b),
            _ => Hit::Miss,
        }
    }

    fn ensure_carried(&mut self) {
        if self.selected != self.carried_index {
            self.carried_index = self.selected;
            self.out.push(C2S::Carried(self.selected));
        }
    }

    /// Minecraft.startAttack.
    fn start_attack(&mut self) {
        if self.miss_time > 0 {
            return;
        }
        match self.hit {
            Hit::Entity(id, _) => {
                // MultiPlayerGameMode.attack: the packet, then the client's own Player.attack.
                self.ensure_carried();
                self.out.push(C2S::Attack { target: id, shift: self.body.keys.shift });
                self.client_attack(id);
                self.attack_strength_ticker = 0;
            }
            Hit::Block(b) if self.level.get(b.pos[0], b.pos[1], b.pos[2]) != Block::Air => {
                self.start_destroy_block(b.pos);
            }
            _ => {
                self.miss_time = 10;
                self.attack_strength_ticker = 0;
            }
        }
        // LocalPlayer.swing sends the swing.
        self.out.push(C2S::Swing(Hand::Main));
    }

    fn same_destroy_target(&self, pos: [i32; 3]) -> bool {
        pos == self.destroy_pos && self.main_hand().same_components(&self.destroying_item)
    }

    /// MultiPlayerGameMode.startDestroyBlock (survival) on bedrock: a new target, which never breaks.
    fn start_destroy_block(&mut self, pos: [i32; 3]) {
        if !self.is_destroying || !self.same_destroy_target(pos) {
            // The packets still go (an abort of the old target, then the start), whatever the block.
            if self.is_destroying {
                self.out.push(C2S::Dig);
            }
            self.out.push(C2S::Dig);
            self.is_destroying = true;
            self.destroy_pos = pos;
            self.destroying_item = self.main_hand();
        }
    }

    /// MultiPlayerGameMode.continueDestroyBlock: the same target goes on (bedrock never breaks), another starts over.
    fn continue_destroy_block(&mut self, pos: [i32; 3]) -> bool {
        self.ensure_carried();
        if !self.same_destroy_target(pos) {
            self.start_destroy_block(pos);
            return true;
        }
        if self.level.get(pos[0], pos[1], pos[2]) == Block::Air {
            self.is_destroying = false;
            return false;
        }
        true
    }

    /// MultiPlayerGameMode.stopDestroyBlock: an abort, and the attack timer starts over.
    fn stop_destroy_block(&mut self) {
        if self.is_destroying {
            self.out.push(C2S::Dig);
            self.is_destroying = false;
            self.attack_strength_ticker = 0;
        }
    }

    /// Minecraft.continueAttack: with attack held, the block in the crosshair keeps breaking (and the arm swings);
    /// otherwise breaking stops.
    fn continue_attack(&mut self, held: bool) {
        if !held {
            self.miss_time = 0;
        }
        if self.miss_time <= 0 {
            match self.hit {
                Hit::Block(b) if held => {
                    if self.level.get(b.pos[0], b.pos[1], b.pos[2]) != Block::Air && self.continue_destroy_block(b.pos) {
                        self.out.push(C2S::Swing(Hand::Main));
                    }
                }
                _ => self.stop_destroy_block(),
            }
        }
    }

    /// The client's Player.attack: only a sprint hit's slowdown shows (Knockback is the server's).
    fn client_attack(&mut self, _id: i32) {
        let delay = attack_delay_for(self.main_hand().item);
        let g = clampf((self.attack_strength_ticker as f32 + 0.5) / delay, 0.0, 1.0);
        self.attack_strength_ticker = 0;
        let bl2 = self.body.sprinting && g > 0.9;
        if bl2 {
            let v = self.body.vel;
            self.body.vel = V3::new(v.x * 0.6, v.y * 1.0, v.z * 0.6);
            self.body.sprinting = false;
            self.body.sprint_mod = false;
        }
    }





    /// Between frames, before the tick: the attack button's mapping (a press sets it, so does a release).
    pub fn between_frames(&mut self, c: &Controls) {
        if c.attacks > 0 || !c.hold_attack {
            self.km_attack = c.hold_attack;
        }
    }









    /// Minecraft.tick for this client: the crosshair, the attack clicks (Minecraft.startAttack) and a held attack
    /// (continueAttack), LocalPlayer.tick, the packets.
    pub fn tick(&mut self, c: &Controls) -> Vec<C2S> {
        self.body.yaw = c.yaw;
        self.body.pitch = c.pitch;
        self.hit = self.pick();
        self.ensure_carried();
        for _ in 0..c.attacks {
            self.start_attack();
        }
        // continueAttack: attack held down.
        self.continue_attack(self.km_attack);
        if self.miss_time > 0 {
            self.miss_time -= 1;
        }
        let keys = c.keys;
        // LocalPlayer.tick: LivingEntity.tick (aiStep), then Player.tick's timer.
        self.old_pos = self.body.pos;
        let level = std::mem::replace(&mut self.level, Level::new([0; 3], [0; 3]));
        let sent = self.body.client_tick(&level, keys);
        self.level = level;
        self.attack_strength_ticker += 1;
        let main = self.main_hand();
        if !(main.item == self.last_item_main.item && main.count == self.last_item_main.count && main.damage == self.last_item_main.damage) {
            if main.item != self.last_item_main.item {
                self.attack_strength_ticker = 0;
            }
            self.last_item_main = main;
        }
        // The input packet, the sprint command, the position.
        if self.last_sent_input != keys {
            self.out.push(C2S::Input(keys));
            self.last_sent_input = keys;
        }
        if let Some(b) = sent.sprint {
            self.out.push(C2S::Sprint(b));
        }
        self.send_position();
        // The other player ticks after this one (it joined later): RemotePlayer.aiStep interpolates, then
        // pushEntities pushes the local player (the only one a client pushes) when the boxes overlap.
        for k in 0..self.entities.len() {
            self.entities[k].interpolate(&self.level);
            let r = self.entities[k];
            if self.health > 0.0 && intersects_bb(&r.bb(), &self.body.bounding_box()) {
                // localPlayer.push(remote).
                let mut d = r.pos.x - self.body.pos.x;
                let mut e = r.pos.z - self.body.pos.z;
                let mut f = d.abs().max(e.abs());
                if f >= 0.01f32 as f64 {
                    f = f.sqrt();
                    d /= f;
                    e /= f;
                    let g = (1.0 / f).min(1.0);
                    d *= g;
                    e *= g;
                    d *= 0.05f32 as f64;
                    e *= 0.05f32 as f64;
                    self.body.vel = self.body.vel.add(V3::new(-d, 0.0, -e));
                }
            }
        }
        // The end of Minecraft.tick.
        self.out.push(C2S::TickEnd);
        std::mem::take(&mut self.out)
    }

    /// LocalPlayer.sendPosition.
    fn send_position(&mut self) {
        let p = self.body.pos;
        let d = p.sub(self.last_pos);
        let g = self.body.yaw as f64 - self.y_rot_last as f64;
        let h = self.body.pitch as f64 - self.x_rot_last as f64;
        self.position_reminder += 1;
        let moved = d.x * d.x + d.y * d.y + d.z * d.z > 2.0e-4 * 2.0e-4 || self.position_reminder >= 20;
        let turned = g != 0.0 || h != 0.0;
        let (og, hc) = (self.body.on_ground, self.body.horizontal_collision);
        if moved || turned || self.last_on_ground != og || self.last_hcol != hc {
            self.out.push(C2S::Move { pos: moved.then_some(p), rot: turned.then_some((self.body.yaw, self.body.pitch)), on_ground: og, hcol: hc });
        }
        if moved {
            self.last_pos = p;
            self.position_reminder = 0;
        }
        if turned {
            self.y_rot_last = self.body.yaw;
            self.x_rot_last = self.body.pitch;
        }
        self.last_on_ground = og;
        self.last_hcol = hc;
    }

    /// One packet from the server.
    pub fn handle(&mut self, s: &S2C) {
        match s {
            S2C::Animate { id, hand } => {
                // LocalPlayer.swing: the client answers its own swing with a swing packet.
                if *id == self.id {
                    self.out.push(C2S::Swing(*hand));
                }
            }
            S2C::Motion { id, v } => {
                if *id == self.id {
                    self.body.vel = lp_vec3(*v);
                    self.motions += 1;
                }
            }
            S2C::Health { hp, food, sat } => {
                self.health = *hp;
                self.food = *food;
                self.saturation = *sat;
                self.body.food = *food;
            }
            S2C::MoveEntity { id, delta, rot, on_ground } => {
                // ClientPacketListener.handleMoveEntity.
                if *id != self.id {
                    if let Some(e) = self.entities.iter_mut().find(|e| e.id == *id) {
                        if let Some(l) = delta {
                            let v = e.decode(*l);
                            e.base = v;
                            match rot {
                                Some((y, x)) => e.interpolate_to(v, unpack_degrees(*y), unpack_degrees(*x)),
                                None => {
                                    let (f, g) = (e.i_yaw(), e.i_pitch());
                                    e.interpolate_to(v, f, g);
                                }
                            }
                        } else if let Some((y, x)) = rot {
                            let v = e.i_pos();
                            e.interpolate_to(v, unpack_degrees(*y), unpack_degrees(*x));
                        }
                        e.on_ground = *on_ground;
                    }
                }
            }
            S2C::PosSync { id, pos, yaw, pitch, on_ground } => {
                // handleEntityPositionSync: interpolated unless it is 64 blocks off.
                if *id != self.id {
                    if let Some(e) = self.entities.iter_mut().find(|e| e.id == *id) {
                        e.base = *pos;
                        if e.pos.distance_to_sqr(*pos) > 4096.0 {
                            e.pos = *pos;
                            e.yaw = *yaw;
                            e.pitch = pitch.clamp(-90.0, 90.0);
                        } else {
                            e.interpolate_to(*pos, *yaw, *pitch);
                        }
                        e.on_ground = *on_ground;
                    }
                }
            }
            // Nothing to roll back: no block ever changes.
            S2C::BlockAck => {}
            S2C::Flags { id, sprint, .. } => {
                if *id == self.id {
                    self.body.sprinting = *sprint;
                }
            }
            S2C::Pose { id, crouch } => {
                if *id == self.id {
                    self.body.crouch_pose = *crouch;
                } else if let Some(e) = self.entities.iter_mut().find(|e| e.id == *id) {
                    e.crouch = *crouch;
                }
            }
            S2C::SprintAttr { id, on } => {
                if *id == self.id {
                    self.body.sprint_mod = *on;
                }
            }
            S2C::Absorption { id, v } => {
                if *id == self.id {
                    self.absorption = *v;
                }
            }
            S2C::PlayerPosition { tid, pos, abs } => {
                // handleMovePlayer: the absolute position, the angles and velocity kept or the angles given
                // and no velocity (set back); confirmed at once, with a move that says not on the ground.
                self.body.pos = *pos;
                match abs {
                    Some((yaw, pitch)) => {
                        self.body.yaw = *yaw;
                        self.body.pitch = pitch.clamp(-90.0, 90.0);
                        self.body.vel = V3::ZERO;
                    }
                    None => {
                        self.body.yaw += 0.0;
                        self.body.pitch = (self.body.pitch + 0.0).clamp(-90.0, 90.0);
                    }
                }
                self.old_pos = *pos;
                self.out.push(C2S::AcceptTeleport(*tid));
                self.out.push(C2S::Move { pos: Some(*pos), rot: Some((self.body.yaw, self.body.pitch)), on_ground: false, hcol: false });
            }
        }
    }
}

/// The attack delay of a held item (Player.getCurrentItemAttackStrengthDelay).
pub fn attack_delay_for(item: Item) -> f32 {
    let mut d = 4.0f64;
    if item.is_sword() {
        d += -2.4f32 as f64;
    }
    (1.0 / d * 20.0) as f32
}

/// Entity.calculateViewVector.
pub fn view_vector(x_rot: f32, y_rot: f32) -> V3 {
    let h = x_rot * DEG;
    let i = -y_rot * DEG;
    let j = mth_cos(i as f64);
    let k = mth_sin(i as f64);
    let l = mth_cos(h as f64);
    let m = mth_sin(h as f64);
    V3::new((k * l) as f64, (-m) as f64, (j * l) as f64)
}

/// The duel: the server, the clients (one per server player that has one), and the packets in flight.
#[derive(Clone, Debug)]
pub struct Duel {
    pub server: Server,
    /// Client for server player i (None: a server-side player without a client).
    pub clients: Vec<Option<Client>>,
    /// Ping in whole ticks on each client's connection (to the server, to the client), as it is when a packet is
    /// sent (it may change between ticks): a packet the other side would handle in tick t handles in tick
    /// t + up (or t + down). One connection keeps its order, so a packet waits for the ones before it.
    pub lag: Vec<(u64, u64)>,
    /// What is on its way, with the tick it arrives in.
    pub up: Vec<VecDeque<(u64, C2S)>>,
    pub down: Vec<VecDeque<(u64, S2C)>>,
    /// The tick being run (counted from 0); between ticks, the next one.
    pub now: u64,
}

impl Duel {
    pub fn new(server: Server, clients: Vec<Option<Client>>) -> Duel {
        let n = clients.len();
        Duel { server, clients, lag: vec![(0, 0); n], up: vec![VecDeque::new(); n], down: vec![VecDeque::new(); n], now: 0 }
    }


    /// One tick. `controls[i]` for each client.
    pub fn tick(&mut self, controls: &[Controls]) {
        // 0. What the clients do between frames goes out at once: the server has it in this tick (plus the lag).
        for i in 0..self.clients.len() {
            if let Some(c) = self.clients[i].as_mut() {
                c.between_frames(&controls[i]);
                let due = self.now + self.lag[i].0;
                self.up[i].extend(c.out.drain(..).map(|pk| (due, pk)));
            }
        }
        // 1. The packets that have reached the server.
        for i in 0..self.up.len() {
            while self.up[i].front().is_some_and(|q| q.0 <= self.now) {
                let (_, pk) = self.up[i].pop_front().unwrap();
                self.server.handle(i, &pk);
            }
        }
        // 2. The server's tick: what it sent (the commands before this tick too) reaches the clients after their
        // own tick (plus the lag).
        self.server.tick();
        self.sent();
        // 3. The clients' ticks: the server has their packets in the next tick (plus the lag).
        for i in 0..self.clients.len() {
            if let Some(c) = self.clients[i].as_mut() {
                let due = self.now + 1 + self.lag[i].0;
                let out = c.tick(&controls[i]);
                self.up[i].extend(out.into_iter().map(|pk| (due, pk)));
            }
        }
        // 4. The clients read what has come, after their own tick.
        self.deliver();
        self.now += 1;
    }

    /// What the server sent so far goes on its way.
    fn sent(&mut self) {
        for i in 0..self.clients.len() {
            let due = self.now + self.lag[i].1;
            self.down[i].extend(self.server.out[i].drain(..).map(|pk| (due, pk)));
        }
    }

    fn deliver(&mut self) {
        for i in 0..self.clients.len() {
            let mut pks = Vec::new();
            while self.down[i].front().is_some_and(|q| q.0 <= self.now) {
                pks.push(self.down[i].pop_front().unwrap().1);
            }
            if let Some(c) = self.clients[i].as_mut() {
                for pk in &pks {
                    c.handle(pk);
                }
                // What the client sent while reading them goes with its tick's packets.
                let due = self.now + 1 + self.lag[i].0;
                self.up[i].extend(c.out.drain(..).map(|pk| (due, pk)));
            }
        }
    }


}


