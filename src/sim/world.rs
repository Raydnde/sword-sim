// SPDX-License-Identifier: AGPL-3.0-only
// sword-sim by jo29_29. Licensed under the GNU AGPL v3 only with additional terms: see LICENSE and ADDITIONAL_TERMS.
//! 1.21.11 player movement, ported line for line from the game's code
//! (LocalPlayer.aiStep/applyInput/modifyInput, KeyboardInput, LivingEntity
//! .aiStep/travel/travelInAir/jumpFromGround, Player.maybeBackOffFromEdge and
//! updatePlayerPose, Entity.move/collide with VoxelShape.collide for full
//! cubes), with Java's float and double semantics kept: the same operations
//! in the same order and widths, Java's Math.min/max/signum on zeros, and
//! Mth's sine table, checked bit for bit against recordings of the real game.
//!
//! The world is full cubes only (bedrock);
//! every block's friction is 0.6 and speed and jump factors 1. No fluids,
//! no climbing, no flying.
//!
//! The server's echo is part of it, as the recordings show it: the packets the
//! client sends at the end of tick N (input, sprint command, position) are
//! handled in the server's step N+1 before its world tick, whose changes to
//! the player's flags byte (the sprint bit) and movement-speed modifiers come
//! back to the client at the end of that step and overwrite its own; the
//! server updates the pose later in its step (Player.tick in doTick), so a
//! pose change reaches the client one step later still.
use std::sync::OnceLock;

/// Mth.SIN: (float) Math.sin(i / 10430.378350470453).
static SIN: OnceLock<Box<[f32]>> = OnceLock::new();
fn sin_table() -> &'static [f32] {
    SIN.get_or_init(|| (0..65536).map(|i| (i as f64 / 10430.378350470453).sin() as f32).collect())
}

pub fn mth_sin(d: f64) -> f32 {
    sin_table()[((d * 10430.378350470453) as i64 & 65535) as usize]
}
pub fn mth_cos(d: f64) -> f32 {
    sin_table()[((d * 10430.378350470453 + 16384.0) as i64 & 65535) as usize]
}

/// Java's Math.min / Math.max for doubles (NaN wins; -0.0 is less than 0.0).
pub fn jmin(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if a == 0. && b == 0. && (a.is_sign_negative() || b.is_sign_negative()) {
        -0.
    } else if a <= b {
        a
    } else {
        b
    }
}
pub fn jmax(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if a == 0. && b == 0. && (a.is_sign_positive() || b.is_sign_positive()) {
        0.
    } else if a >= b {
        a
    } else {
        b
    }
}
/// Java's Math.signum: zeros (and NaN) come back as they are.
pub fn jsignum(d: f64) -> f64 {
    if d == 0. || d.is_nan() {
        d
    } else if d > 0. {
        1.
    } else {
        -1.
    }
}
pub const DEG: f32 = (std::f64::consts::PI / 180.0) as f32;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct V3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
impl V3 {
    pub const ZERO: V3 = V3 { x: 0., y: 0., z: 0. };
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    pub fn add(self, o: V3) -> V3 {
        V3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
    pub fn sub(self, o: V3) -> V3 {
        V3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
    pub fn distance_to_sqr(self, o: V3) -> f64 {
        let (d, e, f) = (o.x - self.x, o.y - self.y, o.z - self.z);
        d * d + e * e + f * f
    }
    pub fn length_sqr(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
    pub fn horizontal_distance_sqr(self) -> f64 {
        self.x * self.x + self.z * self.z
    }
    pub fn scale(self, d: f64) -> V3 {
        V3::new(self.x * d, self.y * d, self.z * d)
    }
    pub fn normalize(self) -> V3 {
        let d = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if d < 1.0e-5f32 as f64 {
            V3::ZERO
        } else {
            V3::new(self.x / d, self.y / d, self.z / d)
        }
    }
    pub fn get(self, axis: usize) -> f64 {
        [self.x, self.y, self.z][axis]
    }
    fn with(self, axis: usize, v: f64) -> V3 {
        let mut a = [self.x, self.y, self.z];
        a[axis] = v;
        V3::new(a[0], a[1], a[2])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}
impl Bb {
    pub fn moved(self, v: V3) -> Bb {
        Bb { min: [self.min[0] + v.x, self.min[1] + v.y, self.min[2] + v.z], max: [self.max[0] + v.x, self.max[1] + v.y, self.max[2] + v.z] }
    }
    /// AABB.expandTowards.
    pub fn expand_towards(self, v: V3) -> Bb {
        let mut b = self;
        for (k, d) in [v.x, v.y, v.z].into_iter().enumerate() {
            if d < 0. {
                b.min[k] += d;
            } else if d > 0. {
                b.max[k] += d;
            }
        }
        b
    }
}

pub trait Cubes {
    /// Whether the block at (x, y, z) is a full cube that collides.
    fn solid(&self, x: i32, y: i32, z: i32) -> bool;
}



/// VoxelShape.collide for the unit cube at `c` (block coordinates) along `axis`.
fn collide_cube(axis: usize, b: &Bb, mut d: f64, c: [i32; 3]) -> f64 {
    if d.abs() < 1.0e-7 {
        return 0.;
    }
    let (a2, a3) = ((axis + 1) % 3, (axis + 2) % 3);
    let lo = |k: usize| c[k] as f64;
    // findIndex on the other two axes: the cube spans them when min + 1e-7 < hi and max - 1e-7 >= lo.
    let spans = |k: usize| b.min[k] + 1.0e-7 < lo(k) + 1. && b.max[k] - 1.0e-7 >= lo(k);
    if !spans(a2) || !spans(a3) {
        return d;
    }
    let (e, f) = (b.max[axis], b.min[axis]);
    if d > 0. {
        if e - 1.0e-7 < lo(axis) {
            let g = lo(axis) - e;
            if g >= -1.0e-7 {
                d = jmin(d, g);
            }
        }
    } else if d < 0. && f + 1.0e-7 >= lo(axis) + 1. {
        let g = lo(axis) + 1. - f;
        if g <= 1.0e-7 {
            d = jmax(d, g);
        }
    }
    d
}

/// The full cubes whose cells meet `b` (BlockCollisions over the box; extra cubes cannot change a result).
pub(crate) fn colliders(w: &impl Cubes, b: &Bb) -> Vec<[i32; 3]> {
    let mut out = Vec::new();
    let lo = |k: usize| (b.min[k] - 1.0e-7).floor() as i32 - 1;
    let hi = |k: usize| (b.max[k] + 1.0e-7).floor() as i32 + 1;
    for x in lo(0)..=hi(0) {
        for y in lo(1)..=hi(1) {
            for z in lo(2)..=hi(2) {
                if w.solid(x, y, z) {
                    out.push([x, y, z]);
                }
            }
        }
    }
    out
}

/// Shapes.collide over the cubes.
fn collide_axis(axis: usize, b: &Bb, cubes: &[[i32; 3]], mut d: f64) -> f64 {
    for &c in cubes {
        if d.abs() < 1.0e-7 {
            return 0.;
        }
        d = collide_cube(axis, b, d, c);
    }
    d
}

/// Entity.collideWithShapes: Y first, then the larger of X and Z last (Direction.axisStepOrder).
pub(crate) fn collide_with_shapes(v: V3, b: &Bb, cubes: &[[i32; 3]]) -> V3 {
    if cubes.is_empty() {
        return v;
    }
    let order = if v.x.abs() < v.z.abs() { [1, 2, 0] } else { [1, 0, 2] };
    let mut out = V3::ZERO;
    for axis in order {
        let d = v.get(axis);
        if d != 0. {
            let e = collide_axis(axis, &b.moved(out), cubes, d);
            out = out.with(axis, e);
        }
    }
    out
}

/// What the client holds this tick (KeyboardInput's keys).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Keys {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub jump: bool,
    pub shift: bool,
    pub sprint: bool,
}

/// The client player's movement state (LocalPlayer and what it inherits).
#[derive(Clone, Debug)]
pub struct Body {
    pub pos: V3,
    pub vel: V3,
    pub yaw: f32,
    pub pitch: f32,
    pub on_ground: bool,
    pub horizontal_collision: bool,
    pub minor_horizontal_collision: bool,
    pub vertical_collision: bool,
    pub vertical_collision_below: bool,
    pub fall_distance: f64,
    pub sprinting: bool,
    /// The sprint modifier on the client's movement speed (LivingEntity.setSprinting sets it with the flag; the
    /// server's attribute echo replaces it on its own).
    pub sprint_mod: bool,
    /// LocalPlayer.crouching (the slowdown) and the pose's (the box).
    pub crouching: bool,
    pub crouch_pose: bool,
    pub no_jump_delay: i32,
    pub jumping: bool,
    pub xxa: f32,
    pub zza: f32,
    pub sprint_trigger_time: i32,
    /// KeyboardInput: last tick's keys and move vector.
    pub keys: Keys,
    pub move_vector: (f32, f32),
    pub sneaking_speed: f64,
    pub main_support: Option<[i32; 3]>,
    pub on_ground_no_blocks: bool,
    pub food: i32,
    /// LocalPlayer.wasSprinting: the sprint state last sent.
    pub was_sprinting: bool,
    /// The server's copy of the player (ServerPlayer), which simulates the same physics with no input from the
    /// position the client last sent: vertical collisions only on a vertical move, no fall distance of its own, no
    /// minor collisions.
    /// The server copy's moves since its last applyEffectsFromBlocks (from, to, the move it was asked for).
    pub moves: Vec<(V3, V3, Option<V3>)>,
    pub server_side: bool,
}

/// What the client sends at the end of its tick besides its position: the sprint command, on a change.
#[derive(Clone, Copy, Debug)]
pub struct Sent {
    pub sprint: Option<bool>,
}

impl Body {
    pub fn new(pos: V3, yaw: f32, pitch: f32) -> Self {
        Self {
            pos,
            vel: V3::ZERO,
            yaw,
            pitch,
            on_ground: false,
            horizontal_collision: false,
            minor_horizontal_collision: false,
            vertical_collision: false,
            vertical_collision_below: false,
            fall_distance: 0.,
            sprinting: false,
            crouching: false,
            crouch_pose: false,
            no_jump_delay: 0,
            jumping: false,
            xxa: 0.,
            zza: 0.,
            sprint_trigger_time: 0,
            keys: Keys::default(),
            move_vector: (0., 0.),
            sneaking_speed: 0.3,
            main_support: None,
            on_ground_no_blocks: false,
            food: 20,
            was_sprinting: false,
            sprint_mod: false,
            moves: Vec::new(),
            server_side: false,
        }
    }

    /// EntityDimensions.makeBoundingBox: half the width as a float, the height as a float.
    pub fn bounding_box_at(&self, pos: V3, crouch: bool) -> Bb {
        let f = (0.6f32 / 2.0f32) as f64;
        let h = if crouch { 1.5f32 } else { 1.8f32 } as f64;
        Bb { min: [pos.x - f, pos.y, pos.z - f], max: [pos.x + f, pos.y + h, pos.z + f] }
    }
    pub fn bounding_box(&self) -> Bb {
        self.bounding_box_at(self.pos, self.crouch_pose)
    }
    pub fn eye_height(&self) -> f32 {
        if self.crouch_pose {
            1.27
        } else {
            1.62
        }
    }

    /// Player's movement speed: 0.1F, times 1.3F while sprinting (ADD_MULTIPLIED_TOTAL), as a float.
    fn speed(&self) -> f32 {
        let mut e = 0.1f32 as f64;
        if self.sprint_mod {
            e *= 1.0 + 0.3f32 as f64;
        }
        e.clamp(0., 1024.) as f32
    }

    fn no_collision(&self, w: &impl Cubes, b: &Bb) -> bool {
        // Level.noCollision: no block shape intersects the box.
        let x0 = b.min[0].floor() as i32;
        let x1 = b.max[0].floor() as i32;
        let y0 = b.min[1].floor() as i32;
        let y1 = b.max[1].floor() as i32;
        let z0 = b.min[2].floor() as i32;
        let z1 = b.max[2].floor() as i32;
        for x in x0..=x1 {
            for y in y0..=y1 {
                for z in z0..=z1 {
                    if w.solid(x, y, z) {
                        let (cx, cy, cz) = (x as f64, y as f64, z as f64);
                        if b.min[0] < cx + 1. && b.max[0] > cx && b.min[1] < cy + 1. && b.max[1] > cy && b.min[2] < cz + 1. && b.max[2] > cz {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
    fn can_fit(&self, w: &impl Cubes, crouch: bool) -> bool {
        let mut b = self.bounding_box_at(self.pos, crouch);
        for k in 0..3 {
            b.min[k] += 1.0e-7;
            b.max[k] -= 1.0e-7;
        }
        self.no_collision(w, &b)
    }

    /// One client tick (LocalPlayer.tick -> Player.tick -> LivingEntity.tick/aiStep -> Player.updatePlayerPose) with
    /// `keys` held; what it sends at the end.
    pub fn client_tick(&mut self, w: &impl Cubes, keys: Keys) -> Sent {
        // LocalPlayer.aiStep.
        if self.sprint_trigger_time > 0 {
            self.sprint_trigger_time -= 1;
        }
        let had_forward = self.move_vector.1 > 1.0e-5;
        let shift_before = self.keys.shift;
        self.crouching = self.can_fit(w, true) && (shift_before || !self.can_fit(w, false));
        // KeyboardInput.tick.
        self.keys = keys;
        let impulse = |a: bool, b: bool| if a == b { 0f32 } else if a { 1. } else { -1. };
        let (mx, my) = (impulse(keys.left, keys.right), impulse(keys.forward, keys.back));
        let len = (mx * mx + my * my).sqrt();
        self.move_vector = if len < 1.0e-4 { (0., 0.) } else { (mx / len, my / len) };
        // LocalPlayer.aiStep: out of any suffocating block, from the box's four corners pulled in to 0.35 width.
        let r = 0.6f32 as f64 * 0.35;
        for (dx, dz) in [(-r, r), (-r, -r), (r, -r), (r, r)] {
            self.move_towards_closest_space(w, self.pos.x + dx, self.pos.z + dz);
        }
        if keys.shift || keys.back {
            self.sprint_trigger_time = 0;
        }
        let forward_now = self.move_vector.1 > 1.0e-5;
        let can_sprint = |b: &Body| b.food > 6;
        let can_start = !self.sprinting && forward_now && can_sprint(self) && !self.crouching;
        if can_start {
            if !had_forward {
                if self.sprint_trigger_time > 0 {
                    self.set_sprinting(true);
                } else {
                    self.sprint_trigger_time = 7;
                }
            }
            if keys.sprint {
                self.set_sprinting(true);
            }
        }
        if self.sprinting && (!can_sprint(self) || !forward_now || self.horizontal_collision && !self.minor_horizontal_collision) {
            self.set_sprinting(false);
        }
        self.living_ai_step(w);
        // Player.tick, after LivingEntity.tick: the pose follows the shift key if it fits.
        let want_crouch = keys.shift;
        self.crouch_pose = if self.can_fit(w, want_crouch) { want_crouch } else if self.can_fit(w, true) { true } else { self.crouch_pose };
        // LocalPlayer.tick: the input packet (on a change), the sprint command (sendIsSprintingIfNeeded), the position.
        let sprint = (self.sprinting != self.was_sprinting).then_some(self.sprinting);
        self.was_sprinting = self.sprinting;
        Sent { sprint }
    }

    /// LocalPlayer.moveTowardsClosestSpace: in a suffocating block at (x, z), 0.1 a tick toward the nearest open side
    /// (west, east, north, south; the first of equals).
    fn move_towards_closest_space(&mut self, w: &impl Cubes, x: f64, z: f64) {
        let (bx, bz) = (x.floor() as i32, z.floor() as i32);
        if !self.suffocates_at(w, bx, bz) {
            return;
        }
        let (f, g) = (x - bx as f64, z - bz as f64);
        // (step x, step z, the distance to that side)
        let sides = [(-1, 0, f), (1, 0, 1. - f), (0, -1, g), (0, 1, 1. - g)];
        let mut best: Option<(i32, i32)> = None;
        let mut h = f64::MAX;
        for (sx, sz, j) in sides {
            if j < h && !self.suffocates_at(w, bx + sx, bz + sz) {
                h = j;
                best = Some((sx, sz));
            }
        }
        if let Some((sx, sz)) = best {
            let v = self.vel;
            self.vel = if sx != 0 { V3::new(0.1 * sx as f64, v.y, v.z) } else { V3::new(v.x, v.y, 0.1 * sz as f64) };
        }
    }

    /// LocalPlayer.suffocatesAt: a suffocating block in the column (bx, bz) over the box's height (deflated 1e-7).
    fn suffocates_at(&self, w: &impl Cubes, bx: i32, bz: i32) -> bool {
        let b = self.bounding_box();
        let (lo, hi) = (b.min[1] + 1.0e-7, b.max[1] - 1.0e-7);
        (lo.floor() as i32..=hi.floor() as i32).any(|y| w.solid(bx, y, bz))
    }

    /// LivingEntity.setSprinting: the flag and the speed modifier.
    fn set_sprinting(&mut self, b: bool) {
        self.sprinting = b;
        self.sprint_mod = b;
    }

    /// LivingEntity.aiStep with LocalPlayer.applyInput.
    fn living_ai_step(&mut self, w: &impl Cubes) {
        if self.no_jump_delay > 0 {
            self.no_jump_delay -= 1;
        }
        let v = self.vel;
        let (mut d, mut e, mut f) = (v.x, v.y, v.z);
        if v.horizontal_distance_sqr() < 9.0e-6 {
            d = 0.;
            f = 0.;
        }
        if e.abs() < 0.003 {
            e = 0.;
        }
        self.vel = V3::new(d, e, f);
        // applyInput -> modifyInput.
        let (mx, my) = self.move_vector;
        let (ix, iy) = if mx * mx + my * my == 0. {
            (mx, my)
        } else {
            let (mut x, mut y) = (mx * 0.98f32, my * 0.98f32);
            if self.crouching {
                let s = self.sneaking_speed as f32;
                x *= s;
                y *= s;
            }
            square_movement(x, y)
        };
        self.xxa = ix;
        self.zza = iy;
        self.jumping = self.keys.jump;
        if self.jumping {
            if self.on_ground && self.no_jump_delay == 0 {
                self.jump_from_ground();
                self.no_jump_delay = 10;
            }
        } else {
            self.no_jump_delay = 0;
        }
        self.travel_in_air(w, V3::new(self.xxa as f64, 0., self.zza as f64));
    }

    /// LivingEntity.aiStep for the server's player: no input (xxa and zza stay 0), never jumping (the server jumps
    /// only from the move packets).
    pub fn server_ai_step(&mut self, w: &impl Cubes) {
        if self.no_jump_delay > 0 {
            self.no_jump_delay -= 1;
        }
        let v = self.vel;
        let (mut d, mut e, mut f) = (v.x, v.y, v.z);
        if v.horizontal_distance_sqr() < 9.0e-6 {
            d = 0.;
            f = 0.;
        }
        if e.abs() < 0.003 {
            e = 0.;
        }
        self.vel = V3::new(d, e, f);
        self.xxa *= 0.98f32;
        self.zza *= 0.98f32;
        self.jumping = false;
        self.no_jump_delay = 0;
        self.travel_in_air(w, V3::new(self.xxa as f64, 0., self.zza as f64));
    }

    /// LivingEntity.jumpFromGround: up to the jump power (0.42F), plus 0.2 along the yaw when sprinting.
    pub fn jump_from_ground(&mut self) {
        let f = 0.42f32 * 1.0f32 * 1.0f32 + 0.;
        let v = self.vel;
        self.vel = V3::new(v.x, jmax(f as f64, v.y), v.z);
        if self.sprinting {
            let g = self.yaw * DEG;
            let add = V3::new((-mth_sin(g as f64)) as f64 * 0.2, 0., mth_cos(g as f64) as f64 * 0.2);
            self.vel = self.vel.add(add);
        }
    }

    /// LivingEntity.travelInAir.
    fn travel_in_air(&mut self, w: &impl Cubes, input: V3) {
        let f: f32 = if self.on_ground { 0.6 } else { 1.0 };
        let g = f * 0.91f32;
        // handleRelativeFrictionAndCalculateMovement: moveRelative, then move.
        let speed = if self.on_ground { self.speed() * (0.216_000_02f32 / (f * f * f)) } else if self.sprinting { 0.025_999_999f32 } else { 0.02f32 };
        self.vel = self.vel.add(input_vector(input, speed, self.yaw));
        let m = self.vel;
        self.move_self(w, m);
        let v = self.vel;
        let d = v.y - 0.08;
        let h = 0.98f32;
        self.vel = V3::new(v.x * g as f64, d * h as f64, v.z * g as f64);
    }







    /// Player.maybeBackOffFromEdge: holding shift on the ground, don't step off an edge more than the step height.
    fn back_off_from_edge(&self, w: &impl Cubes, v: V3) -> V3 {
        let f = 0.6f32;
        if v.y > 0. || !self.keys.shift || !self.is_above_ground(w, f) {
            return v;
        }
        let (mut d, mut e) = (v.x, v.z);
        let h = jsignum(d) * 0.05;
        let i = jsignum(e) * 0.05;
        while d != 0. && self.can_fall_at_least(w, d, 0., f as f64) {
            if d.abs() <= 0.05 {
                d = 0.;
                break;
            }
            d -= h;
        }
        while e != 0. && self.can_fall_at_least(w, 0., e, f as f64) {
            if e.abs() <= 0.05 {
                e = 0.;
                break;
            }
            e -= i;
        }
        while d != 0. && e != 0. && self.can_fall_at_least(w, d, e, f as f64) {
            if d.abs() <= 0.05 {
                d = 0.;
            } else {
                d -= h;
            }
            if e.abs() <= 0.05 {
                e = 0.;
            } else {
                e -= i;
            }
        }
        V3::new(d, v.y, e)
    }
    fn is_above_ground(&self, w: &impl Cubes, f: f32) -> bool {
        self.on_ground || self.fall_distance < f as f64 && !self.can_fall_at_least(w, 0., 0., f as f64 - self.fall_distance)
    }
    fn can_fall_at_least(&self, w: &impl Cubes, d: f64, e: f64, f: f64) -> bool {
        let b = self.bounding_box();
        let probe = Bb { min: [b.min[0] + 1.0e-7 + d, b.min[1] - f - 1.0e-7, b.min[2] + 1.0e-7 + e], max: [b.max[0] - 1.0e-7 + d, b.min[1], b.max[2] - 1.0e-7 + e] };
        self.no_collision(w, &probe)
    }

    /// Entity.move(SELF or PLAYER, v): the same for both.
    pub fn move_self(&mut self, w: &impl Cubes, v: V3) {
        let v = self.back_off_from_edge(w, v);
        let moved = self.collide(w, v);
        let d = moved.length_sqr();
        if d > 1.0e-7 || v.length_sqr() - d < 1.0e-7 {
            // Entity.addMovementThisTick: the server walks these for the blocks the player went through.
            if self.server_side {
                self.moves.push((self.pos, self.pos.add(moved), Some(v)));
            }
            self.pos = self.pos.add(moved);
        }
        let bx = (v.x - moved.x).abs() >= 1.0e-5f32 as f64;
        let bz = (v.z - moved.z).abs() >= 1.0e-5f32 as f64;
        self.horizontal_collision = bx || bz;
        // isLocalInstanceAuthoritative: always for the client's own player, never for the server's copy.
        if !self.server_side || v.y.abs() > 0. {
            self.vertical_collision = v.y != moved.y;
            self.vertical_collision_below = self.vertical_collision && v.y < 0.;
            let below = self.vertical_collision_below;
            let hc = self.horizontal_collision;
            self.set_on_ground_with_movement_h(w, below, hc, moved);
        }
        self.minor_horizontal_collision = self.horizontal_collision && !self.server_side && self.is_horizontal_collision_minor(moved);
        // checkFallDamage (the server's comes from the move packets).
        if !self.server_side {
            if moved.y < 0. {
                self.fall_distance -= moved.y as f32 as f64;
            }
            if self.on_ground {
                self.fall_distance = 0.;
            }
        }
        if self.horizontal_collision {
            let vv = self.vel;
            self.vel = V3::new(if bx { 0. } else { vv.x }, vv.y, if bz { 0. } else { vv.z });
        }
        // Block.updateEntityMovementAfterFallOn: a blocked vertical move stops the vertical motion.
        if v.y != moved.y {
            let vv = self.vel;
            self.vel = V3::new(vv.x * 1.0, vv.y * 0.0, vv.z * 1.0);
        }
        // Block speed factor 1: multiply by (1, 1, 1).
        let vv = self.vel;
        self.vel = V3::new(vv.x * 1.0f32 as f64, vv.y, vv.z * 1.0f32 as f64);
    }

    /// Entity.setOnGroundWithMovement(onGround, horizontalCollision, movement).
    pub fn set_on_ground_with_movement_h(&mut self, w: &impl Cubes, on_ground: bool, horizontal_collision: bool, moved: V3) {
        self.horizontal_collision = horizontal_collision;
        self.set_on_ground_with_movement(w, on_ground, moved);
    }
    fn set_on_ground_with_movement(&mut self, w: &impl Cubes, on_ground: bool, moved: V3) {
        self.on_ground = on_ground;
        // checkSupportingBlock (the supporting block only picks the friction block, 0.6 for all here).
        if on_ground {
            let b = self.bounding_box();
            let probe = Bb { min: [b.min[0], b.min[1] - 1.0e-6, b.min[2]], max: [b.max[0], b.min[1], b.max[2]] };
            let found = support(w, &probe).or_else(|| {
                if self.on_ground_no_blocks {
                    None
                } else {
                    support(w, &probe.moved(V3::new(-moved.x, 0., -moved.z)))
                }
            });
            self.main_support = found;
            self.on_ground_no_blocks = found.is_none();
        } else {
            self.on_ground_no_blocks = false;
            self.main_support = None;
        }
    }

    /// LocalPlayer.isHorizontalCollisionMinor.
    fn is_horizontal_collision_minor(&self, v: V3) -> bool {
        let f = self.yaw * DEG;
        let d = mth_sin(f as f64) as f64;
        let e = mth_cos(f as f64) as f64;
        let g = self.xxa as f64 * e - self.zza as f64 * d;
        let h = self.zza as f64 * e + self.xxa as f64 * d;
        let i = g * g + h * h;
        let j = v.x * v.x + v.z * v.z;
        if !(i < 1.0e-5f32 as f64) && !(j < 1.0e-5f32 as f64) {
            let k = g * v.x + h * v.z;
            let l = (k / (i * j).sqrt()).acos();
            l < 0.139_626_34f32 as f64
        } else {
            false
        }
    }

    /// Entity.collide: the cubes, then the step-up (0.6) when blocked sideways on the ground.
    fn collide(&self, w: &impl Cubes, v: V3) -> V3 {
        let b = self.bounding_box();
        let moved = if v.length_sqr() == 0. { v } else { collide_with_shapes(v, &b, &colliders(w, &b.expand_towards(v))) };
        let bx = v.x != moved.x;
        let by = v.y != moved.y;
        let bz = v.z != moved.z;
        let below = by && v.y < 0.;
        let step = 0.6f32;
        if (below || self.on_ground) && (bx || bz) {
            let b2 = if below { b.moved(V3::new(0., moved.y, 0.)) } else { b };
            let mut b3 = b2.expand_towards(V3::new(v.x, step as f64, v.z));
            if !below {
                b3 = b3.expand_towards(V3::new(0., -1.0e-5f32 as f64, 0.));
            }
            let cubes = colliders(w, &b3);
            let g = moved.y as f32;
            let mut heights: Vec<f32> = Vec::new();
            for c in &cubes {
                for coord in [c[1] as f64, c[1] as f64 + 1.] {
                    let h = (coord - b2.min[1]) as f32;
                    if !(h < 0.) && h != g {
                        if h > step {
                            break;
                        }
                        if !heights.contains(&h) {
                            heights.push(h);
                        }
                    }
                }
            }
            heights.sort_by(|a, b| a.total_cmp(b));
            for hgt in heights {
                let v2 = collide_with_shapes(V3::new(v.x, hgt as f64, v.z), &b2, &cubes);
                if v2.horizontal_distance_sqr() > moved.horizontal_distance_sqr() {
                    let dy = b.min[1] - b2.min[1];
                    return V3::new(v2.x, v2.y - dy, v2.z);
                }
            }
        }
        moved
    }
}

/// Level.findSupportingBlock: the colliding cube nearest the box's centre under it.
fn support(w: &impl Cubes, probe: &Bb) -> Option<[i32; 3]> {
    let mut best: Option<([i32; 3], f64)> = None;
    let center = V3::new((probe.min[0] + probe.max[0]) / 2., probe.min[1], (probe.min[2] + probe.max[2]) / 2.);
    for x in probe.min[0].floor() as i32..=probe.max[0].floor() as i32 {
        for y in probe.min[1].floor() as i32..=probe.max[1].floor() as i32 {
            for z in probe.min[2].floor() as i32..=probe.max[2].floor() as i32 {
                if !w.solid(x, y, z) {
                    continue;
                }
                let (cx, cy, cz) = (x as f64, y as f64, z as f64);
                if !(probe.min[0] < cx + 1. && probe.max[0] > cx && probe.min[1] < cy + 1. && probe.max[1] > cy && probe.min[2] < cz + 1. && probe.max[2] > cz) {
                    continue;
                }
                let dx = cx + 0.5 - center.x;
                let dy = cy + 0.5 - center.y;
                let dz = cz + 0.5 - center.z;
                let d = dx * dx + dy * dy + dz * dz;
                if best.is_none_or(|(_, bd)| d < bd) {
                    best = Some(([x, y, z], d));
                }
            }
        }
    }
    best.map(|b| b.0)
}

/// LocalPlayer.modifyInputSpeedForSquareMovement.
fn square_movement(x: f32, y: f32) -> (f32, f32) {
    let f = (x * x + y * y).sqrt();
    if f <= 0. {
        return (x, y);
    }
    let (ux, uy) = (x * (1.0 / f), y * (1.0 / f));
    let (ax, ay) = (ux.abs(), uy.abs());
    let h = if ay > ax { ax / ay } else { ay / ax };
    let g = (1.0f32 + h * h).sqrt();
    let k = (f * g).min(1.0);
    (ux * k, uy * k)
}

/// Entity.getInputVector.
fn input_vector(v: V3, f: f32, yaw: f32) -> V3 {
    let d = v.length_sqr();
    if d < 1.0e-7 {
        return V3::ZERO;
    }
    let v2 = if d > 1. { v.normalize() } else { v }.scale(f as f64);
    let h = mth_sin((yaw * DEG) as f64);
    let i = mth_cos((yaw * DEG) as f64);
    V3::new(v2.x * i as f64 - v2.z * h as f64, v2.y, v2.z * i as f64 + v2.x * h as f64)
}

