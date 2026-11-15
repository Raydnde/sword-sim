// SPDX-License-Identifier: AGPL-3.0-only
// sword-sim by jo29_29. Licensed under the GNU AGPL v3 only with additional terms: see LICENSE and ADDITIONAL_TERMS.
//! The 1.21.11 server's side of a sword duel, bit for bit (Mojang names, read from the decompiled game; every
//! float and double operation in the game's order and width).
//!
//! What is here: the arena's blocks (air and bedrock), the two ServerPlayers (each with the server's own no-input
//! physics from the position its client last sent, see `Body::server_side`), the damage pipeline (Player.hurtServer,
//! LivingEntity.hurtServer, actuallyHurt: invulnerability, armor with toughness, Protection, absorption),
//! fall damage, suffocation, vanilla FoodData (Normal difficulty), the sword (Player.attack: charge, crits, sprint knockback, durability), and
//! the packets the server handles and sends. `client` is the client and the order of a tick.
use std::sync::Arc;

use crate::sim::world::*;

// ---------------------------------------------------------------- Java helpers

/// Mth.floor.
pub fn floor(d: f64) -> i32 {
    let i = d as i32;
    if d < i as f64 {
        i - 1
    } else {
        i
    }
}
fn lfloor(d: f64) -> i64 {
    let l = d as i64;
    if d < l as f64 {
        l - 1
    } else {
        l
    }
}
/// Mth.frac.
fn frac(d: f64) -> f64 {
    d - lfloor(d) as f64
}
/// Mth.sign (an int).
fn sign(d: f64) -> i32 {
    if d == 0. {
        0
    } else if d > 0. {
        1
    } else {
        -1
    }
}
/// Mth.lerp for doubles.
fn lerp(t: f64, a: f64, b: f64) -> f64 {
    a + t * (b - a)
}
/// Mth.clamp for floats.
pub fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}
/// Java's Math.max / Math.min for floats (NaN wins, -0 below +0).
fn fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        a
    } else if a == 0. && b == 0. {
        if a.is_sign_negative() {
            b
        } else {
            a
        }
    } else if a >= b {
        a
    } else {
        b
    }
}
fn fmin(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        a
    } else if a == 0. && b == 0. {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a <= b {
        a
    } else {
        b
    }
}
/// A velocity after the wire (LpVec3, 1.21.9+): 15 bits per axis, scaled by the
/// largest component rounded up (the in-memory connection still encodes packets).
pub fn lp_vec3(v: V3) -> V3 {
    let san = |d: f64| if d.is_nan() { 0.0 } else { d.clamp(-1.7179869183E10, 1.7179869183E10) };
    let (d, e, f) = (san(v.x), san(v.y), san(v.z));
    let g = d.abs().max(e.abs()).max(f.abs());
    if g < 3.051944088384301E-5 {
        return V3::new(0.0, 0.0, 0.0);
    }
    let l = g.ceil() as i64;
    // Math.round on a non-negative double is f64::round.
    let pack = |x: f64| ((x * 0.5 + 0.5) * 32766.0).round() as i64;
    let unpack = |q: i64| ((q & 32767) as f64).min(32766.0) * 2.0 / 32766.0 - 1.0;
    // The scale goes over the wire as a long (2 bits, then a VarInt of the rest).
    let n = (l & 3) | (((l >> 2) as i32 as i64 & 4294967295) << 2);
    let n = n as f64;
    V3::new(unpack(pack(d / l as f64)) * n, unpack(pack(e / l as f64)) * n, unpack(pack(f / l as f64)) * n)
}
/// Math.round(double), the JDK's exact version.
pub fn java_round(a: f64) -> i64 {
    let bits = a.to_bits() as i64;
    let biased_exp = (bits & 0x7ff0_0000_0000_0000) >> 52;
    let shift = 1074 - biased_exp;
    if shift & -64 == 0 {
        let mut r = (bits & 0x000f_ffff_ffff_ffff) | 0x0010_0000_0000_0000;
        if bits < 0 {
            r = -r;
        }
        ((r >> shift) + 1) >> 1
    } else {
        a as i64
    }
}
/// Mth.packDegrees / unpackDegrees: an angle in a byte.
pub fn pack_degrees(f: f32) -> i8 {
    let g = f * 256.0f32 / 360.0f32;
    let i = g as i32;
    (if g < i as f32 { i - 1 } else { i }) as i8
}
pub fn unpack_degrees(b: i8) -> f32 {
    (b as i32 * 360) as f32 / 256.0f32
}

/// VecDeltaCodec.encode: 1/4096 of a block.
pub fn delta_encode(d: f64) -> i64 {
    java_round(d * 4096.0)
}

/// ServerEntity for a player as the others track it: what they were last told (players go every 2nd tick and send
/// no velocity of their own).
#[derive(Clone, Debug, Default)]
pub struct Track {
    pub tick_count: i32,
    pub base: V3,
    pub yrot: i8,
    pub xrot: i8,
    pub teleport_delay: i32,
    pub was_on_ground: bool,
}




/// Mth.wrapDegrees for floats.
pub fn wrap_degrees(f: f32) -> f32 {
    let mut g = f % 360.0;
    if g >= 180.0 {
        g -= 360.0;
    }
    if g < -180.0 {
        g += 360.0;
    }
    g
}

/// java.util.Random (LegacyRandomSource): each player's dice (Unbreaking).
#[derive(Clone, Debug)]
pub struct JavaRandom {
    seed: u64,
}
impl JavaRandom {
    pub fn new(seed: u64) -> Self {
        JavaRandom { seed: (seed ^ 0x5DEECE66D) & ((1 << 48) - 1) }
    }
    fn next(&mut self, bits: u32) -> i32 {
        self.seed = (self.seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB)) & ((1 << 48) - 1);
        (self.seed >> (48 - bits)) as i64 as i32
    }
    pub fn next_float(&mut self) -> f32 {
        self.next(24) as f32 * 5.9604645e-8f32
    }
}

// ---------------------------------------------------------------- blocks

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Block {
    #[default]
    Air,
    Bedrock,
}
impl Block {
    /// A full collision cube.
    pub fn solid(self) -> bool {
        self == Block::Bedrock
    }
}



/// The arena's blocks: a box of cells; outside it, air, except the flat world's ground (everything below `floor`).
/// The cells are kept in 16-cube sections (and the column bits in 16 x 16 column groups) shared copy-on-write by the
/// levels cloned from one another: the fights on a map share every section nobody changed.
#[derive(Clone, Debug)]
pub struct Level {
    pub origin: [i32; 3],
    pub size: [i32; 3],
    secs: Vec<Arc<[Block]>>,
    /// Sections along x, y and z.
    ns: [u32; 3],
    pub floor: i32,
    /// From this y up, outside the level: solid (a roof).
    pub roof: i32,
    /// Outside the level's columns: solid all the way up (walls).
    pub walled: bool,
    /// Per column (x, z): its solid blocks as bits by y - origin y (empty when the level is over 128 high).
    cols: Vec<Arc<[u128]>>,
}
impl Level {
    pub fn new(origin: [i32; 3], size: [i32; 3]) -> Self {
        let ns = size.map(|n| (n.max(0) as u32).div_ceil(16));
        // Every section starts as the same one (all air), split off when first set; none for an empty level (the
        // placeholder a tick swaps in).
        let n = (ns[0] * ns[1] * ns[2]) as usize;
        let secs = if n > 0 { vec![Arc::from(vec![Block::Air; 4096]); n] } else { Vec::new() };
        let cols = if n > 0 && size[1] <= 128 { vec![Arc::from(vec![0; 256]); (ns[0] * ns[2]) as usize] } else { Vec::new() };
        Level { origin, size, secs, ns, floor: i32::MIN, roof: i32::MAX, walled: false, cols }
    }
    /// A level of `size` from `origin` holding what `f` gives at each cell, built in one go (no copy-on-write per
    /// block); a section is cut short past the level's far sides (a small level is one short section).
    pub fn with(origin: [i32; 3], size: [i32; 3], f: impl Fn(i32, i32, i32) -> Block) -> Self {
        let mut l = Level::new(origin, [0; 3]);
        l.size = size;
        l.ns = size.map(|n| (n.max(0) as u32).div_ceil(16));
        let ns = l.ns;
        let ext = |s: u32, k: usize| (size[k] as u32 - 16 * s).min(16);
        // The column bits, per column group, filled in with the sections.
        let mut cols: Vec<Vec<u128>> = Vec::new();
        if size[1] <= 128 && ns.iter().all(|&n| n > 0) {
            for gz in 0..ns[2] {
                for gx in 0..ns[0] {
                    cols.push(vec![0; ((ext(gz, 2) - 1) * 16 + ext(gx, 0)) as usize]);
                }
            }
        }
        for sy in 0..ns[1] {
            for sz in 0..ns[2] {
                for sx in 0..ns[0] {
                    let (ex, ey, ez) = (ext(sx, 0), ext(sy, 1), ext(sz, 2));
                    let mut sec = vec![Block::Air; ((ey - 1) * 256 + (ez - 1) * 16 + ex) as usize];
                    let g = (sz * ns[0] + sx) as usize;
                    for y in 0..ey {
                        for z in 0..ez {
                            for x in 0..ex {
                                let b = f(origin[0] + (16 * sx + x) as i32, origin[1] + (16 * sy + y) as i32, origin[2] + (16 * sz + z) as i32);
                                if b != Block::Air {
                                    sec[(y << 8 | z << 4 | x) as usize] = b;
                                    if b.solid() && !cols.is_empty() {
                                        cols[g][(z << 4 | x) as usize] |= 1 << (16 * sy + y);
                                    }
                                }
                            }
                        }
                    }
                    l.secs.push(Arc::from(sec));
                }
            }
        }
        l.cols = cols.into_iter().map(Arc::from).collect();
        l
    }
    /// The section and the cell in it.
    #[inline]
    fn index(&self, x: i32, y: i32, z: i32) -> Option<(usize, usize)> {
        let px = x.wrapping_sub(self.origin[0]) as u32;
        let py = y.wrapping_sub(self.origin[1]) as u32;
        let pz = z.wrapping_sub(self.origin[2]) as u32;
        if px >= self.size[0] as u32 || py >= self.size[1] as u32 || pz >= self.size[2] as u32 {
            return None;
        }
        let s = ((py >> 4) * self.ns[2] + (pz >> 4)) * self.ns[0] + (px >> 4);
        Some((s as usize, ((py & 15) << 8 | (pz & 15) << 4 | (px & 15)) as usize))
    }
    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> Block {
        match self.index(x, y, z) {
            // index() keeps both inside: every section holds its cells (a short one up to its last).
            Some((s, i)) => unsafe { *self.secs.get_unchecked(s).get_unchecked(i) },
            None if y < self.floor || y >= self.roof => Block::Bedrock,
            None if self.walled && !self.has_column(x, z) => Block::Bedrock,
            None => Block::Air,
        }
    }
    /// Whether column (x, z) is the level's.
    #[inline]
    pub fn has_column(&self, x: i32, z: i32) -> bool {
        (x.wrapping_sub(self.origin[0]) as u32) < self.size[0] as u32 && (z.wrapping_sub(self.origin[2]) as u32) < self.size[2] as u32
    }
}
impl Cubes for Level {
    fn solid(&self, x: i32, y: i32, z: i32) -> bool {
        self.get(x, y, z).solid()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Face {
    Down,
    Up,
    North,
    South,
    West,
    East,
}
impl Face {
    pub fn step(self) -> [i32; 3] {
        match self {
            Face::Down => [0, -1, 0],
            Face::Up => [0, 1, 0],
            Face::North => [0, 0, -1],
            Face::South => [0, 0, 1],
            Face::West => [-1, 0, 0],
            Face::East => [1, 0, 0],
        }
    }
    fn opposite(self) -> Face {
        match self {
            Face::Down => Face::Up,
            Face::Up => Face::Down,
            Face::North => Face::South,
            Face::South => Face::North,
            Face::West => Face::East,
            Face::East => Face::West,
        }
    }
    /// Direction.getApproximateNearest: the largest dot product, in Direction's order.
    fn approximate_nearest(x: f64, y: f64, z: f64) -> Face {
        let mut best = Face::North;
        let mut g = f32::MIN;
        for f in [Face::Down, Face::Up, Face::North, Face::South, Face::West, Face::East] {
            let s = f.step();
            let h = (x * s[0] as f64 + y * s[1] as f64 + z * s[2] as f64) as f32;
            if h > g {
                g = h;
                best = f;
            }
        }
        best
    }
}

/// A block hit (BlockHitResult): the block, the face, the point, and whether the ray started inside.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockHit {
    pub pos: [i32; 3],
    pub face: Face,
    pub loc: V3,
    pub inside: bool,
}

/// AABB.clipPoint.
#[allow(clippy::too_many_arguments)]
fn clip_point(ds: &mut f64, dir: Option<Face>, d: f64, e: f64, f: f64, g: f64, h: f64, i: f64, j: f64, k: f64, dir2: Face, l: f64, m: f64, n: f64) -> Option<Face> {
    let o = (g - l) / d;
    let p = m + o * e;
    let q = n + o * f;
    if 0.0 < o && o < *ds && h - 1.0e-7 < p && p < i + 1.0e-7 && j - 1.0e-7 < q && q < k + 1.0e-7 {
        *ds = o;
        Some(dir2)
    } else {
        dir
    }
}
/// AABB.getDirection for one box (min d e f, max g h i).
#[allow(clippy::too_many_arguments)]
fn aabb_direction(b: &Bb, from: V3, ds: &mut f64, mut dir: Option<Face>, j: f64, k: f64, l: f64) -> Option<Face> {
    let (d, e, f, g, h, i) = (b.min[0], b.min[1], b.min[2], b.max[0], b.max[1], b.max[2]);
    if j > 1.0e-7 {
        dir = clip_point(ds, dir, j, k, l, d, e, h, f, i, Face::West, from.x, from.y, from.z);
    } else if j < -1.0e-7 {
        dir = clip_point(ds, dir, j, k, l, g, e, h, f, i, Face::East, from.x, from.y, from.z);
    }
    if k > 1.0e-7 {
        dir = clip_point(ds, dir, k, l, j, e, f, i, d, g, Face::Down, from.y, from.z, from.x);
    } else if k < -1.0e-7 {
        dir = clip_point(ds, dir, k, l, j, h, f, i, d, g, Face::Up, from.y, from.z, from.x);
    }
    if l > 1.0e-7 {
        dir = clip_point(ds, dir, l, j, k, f, d, g, e, h, Face::North, from.z, from.x, from.y);
    } else if l < -1.0e-7 {
        dir = clip_point(ds, dir, l, j, k, i, d, g, e, h, Face::South, from.z, from.x, from.y);
    }
    dir
}
/// AABB.clip(from, to) for one box: the point where the segment enters it.
pub fn aabb_clip(b: &Bb, from: V3, to: V3) -> Option<V3> {
    let mut ds = 1.0;
    let (j, k, l) = (to.x - from.x, to.y - from.y, to.z - from.z);
    aabb_direction(b, from, &mut ds, None, j, k, l).map(|_| V3::new(from.x + ds * j, from.y + ds * k, from.z + ds * l))
}
pub fn intersects(a: &Bb, b: &Bb) -> bool {
    a.min[0] < b.max[0] && a.max[0] > b.min[0] && a.min[1] < b.max[1] && a.max[1] > b.min[1] && a.min[2] < b.max[2] && a.max[2] > b.min[2]
}

/// The shape a ray sees in a block: its collision cube.
fn block_shape(b: Block, _outline: bool) -> Option<Bb> {
    b.solid().then_some(Bb { min: [0.; 3], max: [1.; 3] })
}
/// VoxelShape.clip for one of our shapes at `pos`.
fn shape_clip(shape: &Bb, from: V3, to: V3, pos: [i32; 3]) -> Option<BlockHit> {
    let d = to.sub(from);
    if d.length_sqr() < 1.0e-7 {
        return None;
    }
    let p = from.add(d.scale(0.001));
    // CubeVoxelShape.findIndex and isFullWide: the shape's grid is 1 x (1 or 16) x 1.
    let ny = if shape.max[1] < 1.0 { 16.0 } else { 1.0 };
    let idx = |v: f64, n: f64| floor((v * n).clamp(-1.0, n));
    let (ix, iy, iz) = (idx(p.x - pos[0] as f64, 1.0), idx(p.y - pos[1] as f64, ny), idx(p.z - pos[2] as f64, 1.0));
    if ix == 0 && iy == 0 && iz == 0 {
        return Some(BlockHit { pos, face: Face::approximate_nearest(d.x, d.y, d.z).opposite(), loc: p, inside: true });
    }
    let moved = Bb { min: [shape.min[0] + pos[0] as f64, shape.min[1] + pos[1] as f64, shape.min[2] + pos[2] as f64], max: [shape.max[0] + pos[0] as f64, shape.max[1] + pos[1] as f64, shape.max[2] + pos[2] as f64] };
    let mut ds = 1.0;
    let dir = aabb_direction(&moved, from, &mut ds, None, d.x, d.y, d.z)?;
    Some(BlockHit { pos, face: dir, loc: V3::new(from.x + ds * d.x, from.y + ds * d.y, from.z + ds * d.z), inside: false })
}

/// BlockGetter.traverseBlocks: `visit` for each block the segment goes through, in order, until it returns
/// something.
#[inline(always)]
fn traverse<R>(from: V3, to: V3, mut visit: impl FnMut(i32, i32, i32) -> Option<R>) -> Option<R> {
    let d = lerp(-1.0e-7, to.x, from.x);
    let e = lerp(-1.0e-7, to.y, from.y);
    let f = lerp(-1.0e-7, to.z, from.z);
    let g = lerp(-1.0e-7, from.x, to.x);
    let h = lerp(-1.0e-7, from.y, to.y);
    let i = lerp(-1.0e-7, from.z, to.z);
    let (mut j, mut k, mut l) = (floor(g), floor(h), floor(i));
    if let Some(hit) = visit(j, k, l) {
        return Some(hit);
    }
    let (m, n, o) = (d - g, e - h, f - i);
    let (p, q, r) = (sign(m), sign(n), sign(o));
    let s = if p == 0 { f64::MAX } else { p as f64 / m };
    let t = if q == 0 { f64::MAX } else { q as f64 / n };
    let u = if r == 0 { f64::MAX } else { r as f64 / o };
    let mut v = s * if p > 0 { 1.0 - frac(g) } else { frac(g) };
    let mut w = t * if q > 0 { 1.0 - frac(h) } else { frac(h) };
    let mut x = u * if r > 0 { 1.0 - frac(i) } else { frac(i) };
    while v <= 1.0 || w <= 1.0 || x <= 1.0 {
        if v < w {
            if v < x {
                j += p;
                v += s;
            } else {
                l += r;
                x += u;
            }
        } else if w < x {
            k += q;
            w += t;
        } else {
            l += r;
            x += u;
        }
        if let Some(hit) = visit(j, k, l) {
            return Some(hit);
        }
    }
    None
}

/// BlockGetter.clip (traverseBlocks): the first block shape on the segment, or a miss at `to`.
pub fn clip(level: &Level, from: V3, to: V3, outline: bool) -> Result<BlockHit, (V3, Face)> {
    let miss = || {
        let v = from.sub(to);
        (to, Face::approximate_nearest(v.x, v.y, v.z))
    };
    if from == to {
        return Err(miss());
    }
    match traverse(from, to, |x, y, z| block_shape(level.get(x, y, z), outline).and_then(|s| shape_clip(&s, from, to, [x, y, z]))) {
        Some(hit) => Ok(hit),
        None => Err(miss()),
    }
}








// ---------------------------------------------------------------- items

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Item {
    #[default]
    Empty,
    /// The sword kit's: diamond.
    DSword,
    DHelmet,
    DChestplate,
    DLeggings,
    DBoots,
}
impl Item {
    pub fn is_armor(self) -> bool {
        matches!(self, Item::DHelmet | Item::DChestplate | Item::DLeggings | Item::DBoots)
    }
    pub fn is_sword(self) -> bool {
        self == Item::DSword
    }
    pub fn armor_defense(self) -> f64 {
        match self {
            Item::DHelmet | Item::DBoots => 3.0,
            Item::DChestplate => 8.0,
            Item::DLeggings => 6.0,
            _ => 0.0,
        }
    }
    pub fn armor_toughness(self) -> f64 {
        if self.is_armor() { 2.0 } else { 0.0 }
    }
}

/// The enchantment levels the kit uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Ench {
    pub protection: u8,
    pub unbreaking: u8,
    pub sharpness: u8,
    pub knockback: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Stack {
    pub item: Item,
    pub count: i32,
    pub damage: i32,
    pub ench: Ench,
}
impl Stack {
    pub const EMPTY: Stack = Stack { item: Item::Empty, count: 0, damage: 0, ench: Ench { protection: 0, unbreaking: 0, sharpness: 0, knockback: 0 } };
    pub fn is_empty(&self) -> bool {
        self.item == Item::Empty || self.count <= 0
    }
    pub fn max_damage(&self) -> i32 {
        match self.item {
            Item::DHelmet => 363,
            Item::DChestplate => 528,
            Item::DLeggings => 495,
            Item::DBoots => 429,
            Item::DSword => 1561,
            _ => 0,
        }
    }
    /// ItemStack.isSameItemSameComponents.
    pub fn same_components(&self, o: &Stack) -> bool {
        self.item == o.item && self.damage == o.damage && self.ench == o.ench
    }
}


// ---------------------------------------------------------------- the inventory menu


















#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Food {
    pub level: i32,
    pub saturation: f32,
    pub exhaustion: f32,
    pub tick_timer: i32,
}
impl Default for Food {
    fn default() -> Self {
        Food { level: 20, saturation: 5.0, exhaustion: 0.0, tick_timer: 0 }
    }
}
impl Food {
    pub fn add_exhaustion(&mut self, f: f32) {
        self.exhaustion = fmin(self.exhaustion + f, 40.0);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Hand {
    Main,
}

/// The damage types a sword duel has.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DamageKind {
    /// player_attack: armor applies, knockback.
    PlayerAttack,
    Fall,
    Starve,
    /// in_wall (suffocating): bypasses armor, not enchantments.
    InWall,
}
pub struct DamageSource {
    pub kind: DamageKind,
    /// The source position (the attacker's position).
    pub pos: Option<V3>,
    /// The attacking player.
    pub attacker: Option<usize>,
}
impl DamageKind {
    fn bypasses_armor(self) -> bool {
        self != DamageKind::PlayerAttack
    }
    fn no_knockback(self) -> bool {
        self != DamageKind::PlayerAttack
    }
    fn exhaustion(self) -> f32 {
        if self == DamageKind::PlayerAttack { 0.1 } else { 0.0 }
    }
}

// ---------------------------------------------------------------- packets

/// What a client sends.
#[derive(Clone, Debug, PartialEq)]
pub enum C2S {
    /// ServerboundMovePlayerPacket: position and/or rotation, onGround, horizontalCollision.
    Move { pos: Option<V3>, rot: Option<(f32, f32)>, on_ground: bool, hcol: bool },
    Input(Keys),
    Sprint(bool),
    Carried(usize),
    Attack { target: i32, shift: bool },
    Swing(Hand),
    /// ServerboundPlayerActionPacket START/ABORT_DESTROY_BLOCK (on bedrock nothing breaks; only its timing matters).
    Dig,
    /// ServerboundAcceptTeleportationPacket.
    AcceptTeleport(i32),
    /// ServerboundClientTickEndPacket: no move this tick means no known movement.
    TickEnd,
}

/// What the server sends to a client.
#[derive(Clone, Debug, PartialEq)]
pub enum S2C {
    /// ClientboundBlockChangedAckPacket, the answer to a Dig.
    BlockAck,
    /// ClientboundSetEntityMotionPacket.
    Motion { id: i32, v: V3 },
    Health { hp: f32, food: i32, sat: f32 },
    /// Entity data: the shared flags' sprint bit.
    Flags { id: i32, sprint: bool, shift: bool },
    Pose { id: i32, crouch: bool },
    /// The sprint speed modifier (attribute packet).
    SprintAttr { id: i32, on: bool },
    Absorption { id: i32, v: f32 },
    /// ClientboundAnimatePacket: a swing (the player's own client answers its own with a swing packet).
    Animate { id: i32, hand: Hand },
    /// ClientboundMoveEntityPacket (Pos, Rot or PosRot): a position delta in 1/4096 and/or packed angles.
    MoveEntity { id: i32, delta: Option<[i64; 3]>, rot: Option<(i8, i8)>, on_ground: bool },
    /// ClientboundEntityPositionSyncPacket: the exact position and angles.
    PosSync { id: i32, pos: V3, yaw: f32, pitch: f32, on_ground: bool },
    /// ClientboundPlayerPositionPacket: an absolute position; the angles and the velocity
    /// kept (relative, plus nothing), for a move set back (`abs`) the angles given and no velocity.
    PlayerPosition { tid: i32, pos: V3, abs: Option<(f32, f32)> },
}

// ---------------------------------------------------------------- players

/// A ServerPlayer.
#[derive(Clone, Debug)]
pub struct SPlayer {
    pub id: i32,
    /// LivingEntity.yHeadRot: the yaw at the end of its last aiStep.
    pub head_yaw: f32,
    /// The server's physics copy (position from the client, velocity of its own).
    pub body: Body,
    pub first_good: V3,
    pub last_good: V3,
    /// Entity.oldPosition: where the level's entity tick found it.
    pub old_pos: V3,
    pub known_movement: V3,
    pub health: f32,
    pub absorption: f32,
    pub invulnerable_time: i32,
    pub hurt_time: i32,
    pub last_hurt: f32,
    pub food: Food,
    /// A Dig came in: the ack goes in the next connection tick.
    pub ack_due: bool,
    pub inv: [Stack; 36],
    /// Feet, legs, chest, head.
    pub armor: [Stack; 4],
    pub offhand: Stack,
    pub selected: usize,
    pub attack_strength_ticker: i32,
    pub last_item_main: Stack,
    pub input: Keys,
    pub hurt_marked: bool,
    pub dead: bool,
    /// Entity data and attributes to send in the next sendChanges.
    pub flags_dirty: bool,
    pub attrs_dirty: bool,
    pub pose_dirty: bool,
    pub absorption_dirty: bool,
    pub last_sent_health: f32,
    pub last_sent_food: i32,
    pub last_food_saturation_zero: bool,
    /// The inventory as the client last heard it (container sync), the cursor, what the client was last told is
    /// on it, and the menu's state id.
    pub synced: [Stack; 41],
    /// Entity.random.
    pub random: JavaRandom,
    /// LivingEntity.swinging / swingTime (the arm's animation; it decides whether a swing is broadcast).
    pub swinging: bool,
    pub swing_time: i32,
    /// The tracking of this player by the others (ServerEntity), Entity.needsSync, and entity data that changed
    /// (health) beyond the flags above.
    pub track: Track,
    pub needs_sync: bool,
    pub data_dirty: bool,
    /// A teleport the client has not confirmed (its moves and block uses are ignored meanwhile), and its id.
    pub awaiting: Option<V3>,
    pub awaiting_id: i32,
    pub moved_this_tick: bool,
    pub last_equip_main: Stack,
    pub attrs_synced: bool,
    pub current_impulse_impact_pos: Option<V3>,
    pub ignore_fall_damage_from_current_impulse: bool,
    pub current_impulse_context_reset_grace_time: i32,
}



impl SPlayer {
    pub fn new(id: i32, pos: V3, yaw: f32, pitch: f32) -> Self {
        let mut body = Body::new(pos, yaw, pitch);
        body.server_side = true;
        SPlayer {
            id,
            head_yaw: yaw,
            body,
            first_good: pos,
            last_good: pos,
            old_pos: pos,
            known_movement: V3::ZERO,
            health: 20.0,
            absorption: 0.0,
            invulnerable_time: 0,
            hurt_time: 0,
            last_hurt: 0.0,
            food: Food::default(),
            ack_due: false,
            inv: [Stack::EMPTY; 36],
            armor: [Stack::EMPTY; 4],
            offhand: Stack::EMPTY,
            selected: 0,
            attack_strength_ticker: 0,
            last_item_main: Stack::EMPTY,
            input: Keys::default(),
            hurt_marked: false,
            dead: false,
            flags_dirty: false,
            attrs_dirty: false,
            pose_dirty: false,
            absorption_dirty: false,
            last_sent_health: 20.0,
            last_sent_food: 20,
            last_food_saturation_zero: false,
            synced: [Stack::EMPTY; 41],
            random: JavaRandom::new(id as u64),
            swinging: false,
            swing_time: 0,
            track: Track::default(),
            needs_sync: false,
            data_dirty: false,
            awaiting: None,
            awaiting_id: 0,
            moved_this_tick: false,
            last_equip_main: Stack::EMPTY,
            attrs_synced: false,
            current_impulse_impact_pos: None,
            ignore_fall_damage_from_current_impulse: false,
            current_impulse_context_reset_grace_time: 0,
        }
    }
    pub fn try_reset_current_impulse_context(&mut self) {
        if self.current_impulse_context_reset_grace_time == 0 {
            self.reset_current_impulse_context();
        }
    }
    pub fn reset_current_impulse_context(&mut self) {
        self.current_impulse_context_reset_grace_time = 0;
        self.current_impulse_impact_pos = None;
        self.ignore_fall_damage_from_current_impulse = false;
    }

    pub fn pos(&self) -> V3 {
        self.body.pos
    }
    pub fn main_hand(&self) -> Stack {
        self.inv[self.selected]
    }
    pub fn slot(&self, i: usize) -> Stack {
        match i {
            0..=35 => self.inv[i],
            36..=39 => self.armor[i - 36],
            _ => self.offhand,
        }
    }



    pub fn eye_y(&self) -> f64 {
        self.body.pos.y + self.body.eye_height() as f64
    }
    pub fn eye(&self) -> V3 {
        V3::new(self.body.pos.x, self.eye_y(), self.body.pos.z)
    }
    pub fn bb(&self) -> Bb {
        self.body.bounding_box()
    }
    pub fn is_hurt(&self) -> bool {
        self.health > 0.0 && self.health < 20.0
    }
    /// Attributes.MAX_ABSORPTION: 0 (no Absorption effect in a sword duel).
    pub fn max_absorption(&self) -> f32 {
        0.0
    }
    /// LivingEntity.setAbsorptionAmount: clamped to [0, max].
    pub fn set_absorption(&mut self, f: f32) {
        let v = clampf(f, 0.0, self.max_absorption());
        if v != self.absorption {
            self.absorption_dirty = true;
        }
        self.absorption = v;
    }
    pub fn set_health(&mut self, f: f32) {
        let v = clampf(f, 0.0, 20.0);
        if v.to_bits() != self.health.to_bits() {
            self.data_dirty = true;
        }
        self.health = v;
    }
    pub fn heal(&mut self, f: f32) {
        let g = self.health;
        if g > 0.0 {
            self.set_health(g + f);
        }
    }
    fn armor_pieces(&self) -> impl Iterator<Item = &Stack> {
        self.armor.iter().filter(|s| !s.is_empty() && s.item.is_armor())
    }
    /// getArmorValue (floor of the attribute) and the toughness as a float.
    pub fn armor_value(&self) -> (i32, f32) {
        let mut a = 0.0f64;
        let mut t = 0.0f64;
        for s in self.armor_pieces() {
            a += s.item.armor_defense();
            t += s.item.armor_toughness();
        }
        (floor(a.clamp(0.0, 30.0)), t.clamp(0.0, 20.0) as f32)
    }
    /// Attributes.KNOCKBACK_RESISTANCE: diamond armor gives none.
    pub fn knockback_resistance(&self) -> f64 {
        0.0
    }
    /// Attributes.SNEAKING_SPEED: 0.3.
    pub fn sneaking_speed(&self) -> f64 {
        0.3f64
    }
    /// EnchantmentHelper.getDamageProtection: Protection, 1 a level.
    pub fn damage_protection(&self, _kind: DamageKind) -> f32 {
        let mut f = 0.0f32;
        for s in self.armor_pieces() {
            let e = s.ench;
            if e.protection > 0 {
                f += 1.0 + 1.0 * (e.protection as f32 - 1.0);
            }
        }
        f
    }
    /// Attributes.ATTACK_DAMAGE and ATTACK_SPEED for a specific main-hand item.
    pub fn attack_damage_for(&self, item: Item) -> f32 {
        let mut base = 1.0f64;
        if item == Item::DSword {
            base += 6.0f32 as f64;
        }
        base.clamp(0.0, 2048.0) as f32
    }
    pub fn attack_speed_for(&self, item: Item) -> f64 {
        let mut d = 4.0f64;
        if item.is_sword() {
            d += -2.4f32 as f64;
        }
        d.clamp(0.0, 1024.0)
    }
    /// Player.getCurrentItemAttackStrengthDelay.
    pub fn attack_delay_for(&self, item: Item) -> f32 {
        (1.0 / self.attack_speed_for(item) * 20.0) as f32
    }
    /// Player.getAttackStrengthScale.
    pub fn attack_strength_scale_for(&self, item: Item, f: f32) -> f32 {
        clampf((self.attack_strength_ticker as f32 + f) / self.attack_delay_for(item), 0.0, 1.0)
    }
    pub fn reset_attack_strength(&mut self) {
        self.attack_strength_ticker = 0;
    }
}













/// A player's tallies over a fight, for `info` only (nothing in the game reads them): damage taken in
/// all, from the other player and from itself (health and absorption lost),
/// hits landed.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub taken: f32,
    pub by_other: f32,
    pub by_self: f32,
}


/// The server: the level, the players and what goes to each client.
#[derive(Clone, Debug)]
pub struct Server {
    pub stats: [Stats; 2],
    pub level: Level,
    pub players: Vec<SPlayer>,
    pub next_id: i32,
    pub tick_count: i64,
    pub random: JavaRandom,
    /// Packets to each player's client, in order.
    pub out: Vec<Vec<S2C>>,
}

impl Server {
    pub fn new(level: Level) -> Self {
        Server {
            level,
            stats: Default::default(),
            players: Vec::new(),
            next_id: 1,
            tick_count: 0,
            random: JavaRandom::new(0),
            out: Vec::new(),
        }
    }
    pub fn send(&mut self, p: usize, s: S2C) {
        if p < self.out.len() {
            self.out[p].push(s);
        }
    }
    fn send_all(&mut self, s: S2C) {
        for p in 0..self.out.len() {
            self.out[p].push(s.clone());
        }
    }

    /// LivingEntity.swing(hand, toSelf): the animation, broadcast unless the arm is early in a swing.
    pub fn swing(&mut self, i: usize, hand: Hand, to_self: bool) {
        let p = &mut self.players[i];
        // getCurrentSwingDuration: 6 for everything in the kit (no Haste or Mining Fatigue).
        if !p.swinging || p.swing_time >= 6 / 2 || p.swing_time < 0 {
            p.swing_time = -1;
            p.swinging = true;
            let id = p.id;
            for o in 0..self.players.len() {
                if o != i || to_self {
                    self.send(o, S2C::Animate { id, hand });
                }
            }
        }
    }

    // ------------------------------------------------------------ damage

    /// Player.hurtServer then LivingEntity.hurtServer, counted into `stats`.
    pub fn hurt(&mut self, i: usize, src: &DamageSource, f: f32) -> bool {
        let hp0 = self.players[i].health + self.players[i].absorption;
        let r = self.hurt_inner(i, src, f);
        if i < 2 {
            let p = &self.players[i];
            let lost = if p.dead { hp0 } else { (hp0 - p.health - p.absorption).max(0.0) };
            let st = &mut self.stats[i];
            st.taken += lost;
            match src.attacker {
                Some(a) if a == i => st.by_self += lost,
                Some(_) => st.by_other += lost,
                None => {}
            }
        }
        r
    }

    fn hurt_inner(&mut self, i: usize, src: &DamageSource, mut f: f32) -> bool {
        if self.players[i].dead || self.players[i].health <= 0.0 {
            return false;
        }
        if f == 0.0 {
            return false;
        }
        // LivingEntity.hurtServer (no shield to block with).
        if f < 0.0 {
            f = 0.0;
        }
        if f.is_nan() || f.is_infinite() {
            f = f32::MAX;
        }
        let mut bl2 = true;
        {
            let p = &self.players[i];
            if p.invulnerable_time as f32 > 10.0 {
                if f <= p.last_hurt {
                    return false;
                }
                let lh = p.last_hurt;
                self.actually_hurt(i, src, f - lh);
                self.players[i].last_hurt = f;
                bl2 = false;
            } else {
                self.players[i].last_hurt = f;
                self.players[i].invulnerable_time = 20;
                self.actually_hurt(i, src, f);
                self.players[i].hurt_time = 10;
            }
        }
        if bl2 {
            // markHurt.
            self.players[i].hurt_marked = true;
            if !src.kind.no_knockback() {
                let (mut d, mut e) = (0.0, 0.0);
                if let Some(sp) = src.pos {
                    d = sp.x - self.players[i].pos().x;
                    e = sp.z - self.players[i].pos().z;
                }
                self.knockback(i, 0.4f32 as f64, d, e);
            }
        }
        if self.players[i].health <= 0.0 {
            self.players[i].dead = true;
        }
        true
    }

    /// Player.actuallyHurt.
    fn actually_hurt(&mut self, i: usize, src: &DamageSource, mut f: f32) {
        // getDamageAfterArmorAbsorb.
        if !src.kind.bypasses_armor() {
            self.hurt_armor(i, f);
            let (armor, toughness) = self.players[i].armor_value();
            f = after_armor(f, armor as f32, toughness);
        }
        // getDamageAfterMagicAbsorb (no Resistance effect here).
        let p = &self.players[i];
        if f <= 0.0 {
            f = 0.0;
        } else {
            let l = p.damage_protection(src.kind);
            if l > 0.0 {
                let h = clampf(l, 0.0, 20.0);
                f = f * (1.0 - h / 25.0);
            }
        }
        let p = &mut self.players[i];
        let var8 = fmax(f - p.absorption, 0.0);
        let abs = p.absorption;
        p.set_absorption(abs - (f - var8));
        if var8 != 0.0 {
            p.food.add_exhaustion(src.kind.exhaustion());
            let h = p.health;
            p.set_health(h - var8);
        }
    }

    /// Player.hurtArmor -> doHurtEquipment: max(1, f / 4) durability a piece, less what Unbreaking saves.
    fn hurt_armor(&mut self, i: usize, f: f32) {
        if f <= 0.0 {
            return;
        }
        let n = fmax(1.0, f / 4.0) as i32;
        for slot in 0..4 {
            let s = self.players[i].armor[slot];
            if s.is_empty() || !s.item.is_armor() {
                continue;
            }
            let amount = self.unbreaking(s, n);
            let p = &mut self.players[i];
            p.armor[slot].damage += amount;
            if p.armor[slot].damage >= p.armor[slot].max_damage() {
                p.armor[slot] = Stack::EMPTY;
            }
        }
    }
    /// EnchantmentHelper.processDurabilityChange: Unbreaking's remove_binomial.
    fn unbreaking(&mut self, s: Stack, n: i32) -> i32 {
        let l = s.ench.unbreaking as f32;
        if l <= 0.0 {
            return n;
        }
        let chance = if s.item.is_armor() { (2.0 + 2.0 * (l - 1.0)) / (10.0 + 5.0 * (l - 1.0)) } else { (1.0 + (l - 1.0)) / (2.0 + (l - 1.0)) };
        let mut removed = 0;
        for _ in 0..n {
            if self.random.next_float() < chance {
                removed += 1;
            }
        }
        n - removed
    }

    /// LivingEntity.knockback.
    fn knockback(&mut self, i: usize, d: f64, mut e: f64, mut f: f64) {
        let p = &mut self.players[i];
        let d = d * (1.0 - p.knockback_resistance());
        if d <= 0.0 {
            return;
        }
        let v = p.body.vel;
        let mut tries = 0;
        while e * e + f * f < 1.0e-5f32 as f64 && tries < 100 {
            // The entity's random: a tiny random direction (only when the attacker stands exactly on the target).
            e = 0.001;
            f = 0.0;
            tries += 1;
        }
        p.needs_sync = true;
        let k = V3::new(e, 0.0, f).normalize().scale(d);
        p.body.vel = V3::new(v.x / 2.0 - k.x, if p.body.on_ground { jmin(0.4, v.y / 2.0 + d) } else { v.y }, v.z / 2.0 - k.z);
    }

    // ------------------------------------------------------------ packets

    /// Handle one packet from player i's client.
    pub fn handle(&mut self, i: usize, pk: &C2S) {
        match pk {
            C2S::Move { pos, rot, on_ground, hcol } => {
                let p = &mut self.players[i];
                if p.awaiting.is_some() {
                    // updateAwaitingTeleport: only the rotation counts until the teleport is confirmed.
                    if let Some((y, x)) = rot {
                        p.body.yaw = wrap_degrees(*y) % 360.0;
                        p.body.pitch = wrap_degrees(*x).clamp(-90.0, 90.0) % 360.0;
                    }
                } else {
                    self.handle_move(i, *pos, *rot, *on_ground, *hcol);
                }
            }
            C2S::AcceptTeleport(id) => {
                let p = &mut self.players[i];
                if *id == p.awaiting_id {
                    if let Some(a) = p.awaiting.take() {
                        p.body.pos = a;
                        p.last_good = a;
                    }
                }
            }
            C2S::TickEnd => {
                let p = &mut self.players[i];
                if !p.moved_this_tick {
                    p.known_movement = V3::ZERO;
                }
                p.moved_this_tick = false;
            }
            C2S::Input(k) => {
                let p = &mut self.players[i];
                let _just_jumped = k.jump && !p.body.keys.jump;
                p.input = *k;
                if p.body.keys.shift != k.shift {
                    p.flags_dirty = true;
                }
                p.body.keys.shift = k.shift;
                p.body.keys.jump = k.jump;
                p.body.keys.forward = k.forward;
                p.body.keys.back = k.back;
                p.body.keys.left = k.left;
                p.body.keys.right = k.right;
                p.body.keys.sprint = k.sprint;
            }
            C2S::Sprint(b) => {
                let p = &mut self.players[i];
                if p.body.sprinting != *b {
                    p.flags_dirty = true;
                    p.attrs_dirty = true;
                }
                p.body.sprinting = *b;
                p.body.sprint_mod = *b;
            }
            C2S::Carried(s) => {
                let p = &mut self.players[i];
                if *s < 9 {
                    p.selected = *s;
                }
            }
            C2S::Attack { target, shift } => {
                {
                    let p = &mut self.players[i];
                    if p.body.keys.shift != *shift {
                        p.flags_dirty = true;
                    }
                    p.body.keys.shift = *shift;
                }
                self.handle_attack(i, *target);
            }
            C2S::Dig => self.players[i].ack_due = true,
            C2S::Swing(h) => {
                // ServerPlayer.swing: the animation for the others, and the attack timer starts over.
                self.swing(i, *h, false);
                self.players[i].reset_attack_strength();
            }
        }
    }

    /// ServerGamePacketListenerImpl.handleMovePlayer (no teleports pending, no vehicles).
    fn handle_move(&mut self, i: usize, pos: Option<V3>, rot: Option<(f32, f32)>, on_ground: bool, hcol: bool) {
        let w = &self.level;
        let p = &mut self.players[i];
        let (f, g) = match rot {
            Some((y, x)) => (wrap_degrees(y), wrap_degrees(x)),
            None => (wrap_degrees(p.body.yaw), wrap_degrees(p.body.pitch)),
        };
        let target = pos.unwrap_or(p.body.pos);
        let (d, e, h) = (target.x.clamp(-3.0e7, 3.0e7), target.y.clamp(-2.0e7, 2.0e7), target.z.clamp(-3.0e7, 3.0e7));
        let (oi, oj, ok) = (p.body.pos.x, p.body.pos.y, p.body.pos.z);
        // (The "moved too quickly" check never fires in a duel.)
        let old_bb = p.body.bounding_box();
        let l = d - p.last_good.x;
        let m = e - p.last_good.y;
        let n = h - p.last_good.z;
        let bl2 = m > 0.0;
        if p.body.on_ground && !on_ground && bl2 {
            // ServerPlayer.jumpFromGround: the jump costs food too.
            p.body.jump_from_ground();
            p.needs_sync = true;
            let f = if p.body.sprinting { 0.2f32 } else { 0.05f32 };
            p.food.add_exhaustion(f);
        }
        p.body.move_self(w, V3::new(l, m, n));
        // "Moved wrongly": our move ends more than 0.25 from where the client says (sideways only: the height never
        // counts, `m > -0.5 || m < 0.5`). Wrongly from a free spot, or into a block it was not already in, and the
        // player is set back where it was (a client that has not heard of a block yet walks into it).
        let (wx, wz) = (d - p.body.pos.x, h - p.body.pos.z);
        let wrong = wx * wx + wz * wz > 0.0625;
        let new_bb = p.body.bounding_box().moved(V3::new(d - p.body.pos.x, e - p.body.pos.y, h - p.body.pos.z));
        if wrong && no_collision(w, &old_bb) || colliding_with_anything_new(w, &old_bb, &new_bb) {
            // ServerGamePacketListenerImpl.teleport(i, j, k, f, g): absolute, no velocity.
            let back = V3::new(oi, oj, ok);
            p.awaiting_id += 1;
            p.body.pos = back;
            p.body.yaw = f;
            p.body.pitch = g.clamp(-90.0, 90.0);
            p.head_yaw = f;
            p.old_pos = back;
            p.body.moves.clear();
            p.body.vel = V3::ZERO;
            p.awaiting = Some(back);
            let tid = p.awaiting_id;
            let (yaw, pitch) = (p.body.yaw, p.body.pitch);
            self.send(i, S2C::PlayerPosition { tid, pos: back, abs: Some((yaw, pitch)) });
            // doCheckFallDamage with no movement.
            let p = &mut self.players[i];
            let fall = p.body.fall_distance;
            if on_ground {
                p.body.fall_distance = 0.0;
                if fall > 0.0 {
                    self.check_player_fall_damage(i, fall);
                }
                self.players[i].body.fall_distance = 0.0;
            }
            return;
        }
        p.body.pos = V3::new(d, e, h);
        p.body.yaw = f;
        p.body.pitch = g;
        let moved = V3::new(d - oi, e - oj, h - ok);
        p.body.set_on_ground_with_movement_h(w, on_ground, hcol, moved);
        // doCheckFallDamage -> checkFallDamage with the movement.
        if moved.y < 0.0 {
            p.body.fall_distance -= moved.y as f32 as f64;
        }
        let fall = p.body.fall_distance;
        if on_ground {
            if fall > 0.0 {
                self.check_player_fall_damage(i, fall);
            }
            self.players[i].body.fall_distance = 0.0;
        }
        let p = &mut self.players[i];
        p.known_movement = moved;
        p.moved_this_tick = true;
        if bl2 {
            p.body.fall_distance = 0.0;
        }
        // checkMovementStatistics: sprinting on the ground costs 0.1 a metre (walking and crouching nothing).
        if !(moved.x == 0.0 && moved.y == 0.0 && moved.z == 0.0) && p.body.on_ground && p.body.sprinting {
            let cm = ((moved.x * moved.x + moved.z * moved.z).sqrt() as f32 * 100.0f32) as f64;
            let k = (cm + 0.5).floor() as i32;
            if k > 0 {
                p.food.add_exhaustion(0.1f32 * k as f32 * 0.01f32);
            }
        }
        p.last_good = p.body.pos;
    }

    fn check_player_fall_damage(&mut self, i: usize, fall: f64) {
        let p = &mut self.players[i];
        let bl = p.current_impulse_impact_pos.is_some() && p.ignore_fall_damage_from_current_impulse;
        let eff_fall = if bl {
            let e = fall.min(p.current_impulse_impact_pos.unwrap().y - p.body.pos.y);
            if e <= 0.0 {
                p.reset_current_impulse_context();
            } else {
                p.try_reset_current_impulse_context();
            }
            e
        } else {
            fall
        };
        p.body.fall_distance = 0.0;
        if eff_fall > 0.0 {
            let dmg = fall_damage(eff_fall);
            if dmg > 0 {
                self.players[i].reset_current_impulse_context();
                self.hurt(i, &DamageSource { kind: DamageKind::Fall, pos: None, attacker: None }, dmg as f32);
            }
        }
        self.players[i].body.fall_distance = 0.0;
    }

    /// handleInteract (attack): reach, then Player.attack.
    fn handle_attack(&mut self, i: usize, target: i32) {
        let eye = self.players[i].eye();
        let Some(bb) = self.players.iter().find(|p| p.id == target && !p.dead).map(|p| p.bb()) else { return };
        // isWithinAttackRange: 3.0, plus the server's tolerance of 3.0.
        let r = 3.0 + 3.0;
        if bb_distance_sqr(&bb, eye) >= r * r {
            return;
        }
        if self.players.iter().position(|p| p.id == target) == Some(i) {
            return;
        }
        self.attack(i, target);
    }

    /// Player.attack on the other player (the server's: Sharpness and Knockback count here).
    pub fn attack(&mut self, i: usize, target: i32) {
        let Some(t) = self.players.iter().position(|p| p.id == target) else { return };
        let p = &self.players[i];
        let weapon = p.main_hand();
        let attr_item = if p.attrs_synced { p.last_equip_main.item } else { weapon.item };
        let mut f = p.attack_damage_for(attr_item);
        let g = p.attack_strength_scale_for(attr_item, 0.5);
        let sharp = if weapon.item.is_sword() { weapon.ench.sharpness as f32 } else { 0.0 };
        let enchanted = if sharp > 0.0 { f + (1.0 + 0.5 * (sharp - 1.0)) } else { f };
        let h = g * (enchanted - f);
        f *= 0.2 + g * g * 0.8;
        // onAttack: resetOnlyAttackStrengthTicker.
        self.players[i].attack_strength_ticker = 0;
        if !(f > 0.0 || h > 0.0) {
            return;
        }
        let p = &self.players[i];
        let bl = g > 0.9;
        let bl2 = p.body.sprinting && bl;
        let crit = bl && p.body.fall_distance > 0.0 && !p.body.on_ground && !p.body.sprinting;
        if crit {
            f *= 1.5;
        }
        let total = f + h;
        let yaw = p.body.yaw;
        let attacker_pos = p.pos();
        let target_vel = self.players[t].body.vel;
        let src = DamageSource { kind: DamageKind::PlayerAttack, pos: Some(attacker_pos), attacker: Some(i) };
        if !self.hurt(t, &src, total) {
            return;
        }
        // causeExtraKnockback: Knockback's 1 a level, halved, plus 0.5 for a sprint hit.
        let kb = weapon.ench.knockback as f32;
        let k = (if kb > 0.0 { 1.0 + 1.0 * (kb - 1.0) } else { 0.0 }) / 2.0 + if bl2 { 0.5 } else { 0.0 };
        if k > 0.0 {
            let r = yaw * DEG;
            self.knockback(t, k as f64, mth_sin(r as f64) as f64, -mth_cos(r as f64) as f64);
            let a = &mut self.players[i];
            a.body.vel = V3::new(a.body.vel.x * 0.6, a.body.vel.y * 1.0, a.body.vel.z * 0.6);
            if a.body.sprinting {
                a.flags_dirty = true;
                a.attrs_dirty = true;
            }
            a.body.sprinting = false;
            a.body.sprint_mod = false;
        }
        if self.players[t].hurt_marked {
            let id = self.players[t].id;
            let v = self.players[t].body.vel;
            self.send_all(S2C::Motion { id, v });
            self.players[t].hurt_marked = false;
            self.players[t].body.vel = target_vel;
        }
        // (isSweepAttack: with only the two players there is nobody else for the sweep to reach.)
        if weapon.item.is_sword() {
            // hurtEnemy: the sword loses 1 (less Unbreaking).
            let n = self.unbreaking(weapon, 1);
            let a = &mut self.players[i];
            a.inv[a.selected].damage += n;
        }
        self.players[i].food.add_exhaustion(0.1);
    }





    // ------------------------------------------------------------ the tick


    /// The server's tick: the level (sendChanges first, then the entities), then each connection (doTick).
    pub fn tick(&mut self) {
        // ServerLevel.tick: tickTime, the block ticks, then the chunk source (ChunkMap.tick -> sendChanges).
        self.tick_count += 1;
        for i in 0..self.players.len() {
            self.send_changes(i);
        }
        // The entities: setOldPosAndRot, then ServerPlayer.tick (its invulnerability, broadcastChanges).
        for i in 0..self.players.len() {
            self.players[i].old_pos = self.players[i].body.pos;
            let p = &mut self.players[i];
            if p.invulnerable_time > 0 {
                p.invulnerable_time -= 1;
            }
        }
        // The connections: ServerGamePacketListenerImpl.tick -> tickPlayer -> doTick.
        for i in 0..self.players.len() {
            if std::mem::take(&mut self.players[i].ack_due) {
                self.send(i, S2C::BlockAck);
            }
            self.do_tick(i);
        }
    }


    /// ServerEntity.sendChanges for player i: to those tracking it, the position and angles every 2nd tick (or
    /// whenever it needs a sync or its entity data changed); to them and itself, the entity data, the attributes,
    /// and the knockback of a hit.
    fn send_changes(&mut self, i: usize) {
        let p = &mut self.players[i];
        let id = p.id;
        let mut out = Vec::new();
        let mut others = Vec::new();
        let dirty = p.flags_dirty || p.pose_dirty || p.absorption_dirty || p.data_dirty;
        if p.track.tick_count % 2 == 0 || p.needs_sync || dirty {
            let t = &mut p.track;
            let b = pack_degrees(p.body.yaw);
            let c = pack_degrees(p.body.pitch);
            let rot = (b as i32 - t.yrot as i32).abs() >= 1 || (c as i32 - t.xrot as i32).abs() >= 1;
            t.teleport_delay += 1;
            let pos = p.body.pos;
            let dv = pos.sub(t.base);
            let moved = dv.x * dv.x + dv.y * dv.y + dv.z * dv.z >= 7.6293945e-6f32 as f64;
            let bl3 = moved || t.tick_count % 60 == 0;
            let l = [delta_encode(pos.x) - delta_encode(t.base.x), delta_encode(pos.y) - delta_encode(t.base.y), delta_encode(pos.z) - delta_encode(t.base.z)];
            let far = l.iter().any(|v| *v < -32768 || *v > 32767);
            let on_ground = p.body.on_ground;
            let (mut b4, mut b5) = (false, false);
            let pk = if far || t.teleport_delay > 400 || t.was_on_ground != on_ground {
                t.was_on_ground = on_ground;
                t.teleport_delay = 0;
                b4 = true;
                b5 = true;
                Some(S2C::PosSync { id, pos, yaw: p.body.yaw, pitch: p.body.pitch, on_ground })
            } else if !(bl3 && rot) {
                if bl3 {
                    b4 = true;
                    Some(S2C::MoveEntity { id, delta: Some(l), rot: None, on_ground })
                } else if rot {
                    b5 = true;
                    Some(S2C::MoveEntity { id, delta: None, rot: Some((b, c)), on_ground })
                } else {
                    None
                }
            } else {
                b4 = true;
                b5 = true;
                Some(S2C::MoveEntity { id, delta: Some(l), rot: Some((b, c)), on_ground })
            };
            others.extend(pk);
            // sendDirtyEntityData: the data, then the attributes.
            if p.flags_dirty {
                out.push(S2C::Flags { id, sprint: p.body.sprinting, shift: p.body.keys.shift });
            }
            if p.pose_dirty {
                out.push(S2C::Pose { id, crouch: p.body.crouch_pose });
            }
            if p.absorption_dirty {
                out.push(S2C::Absorption { id, v: p.absorption });
            }
            p.flags_dirty = false;
            p.pose_dirty = false;
            p.absorption_dirty = false;
            p.data_dirty = false;
            if p.attrs_dirty {
                out.push(S2C::SprintAttr { id, on: p.body.sprint_mod });
                p.attrs_dirty = false;
            }
            let t = &mut p.track;
            if b4 {
                t.base = pos;
            }
            if b5 {
                t.yrot = b;
                t.xrot = c;
            }
            p.needs_sync = false;
        }
        p.track.tick_count += 1;
        if p.hurt_marked {
            p.hurt_marked = false;
            out.push(S2C::Motion { id, v: p.body.vel });
        }
        for s in others {
            for j in 0..self.out.len() {
                if j != i {
                    self.out[j].push(s.clone());
                }
            }
        }
        // To the player and to those tracking it.
        for s in out {
            self.send_all(s);
        }
    }

    /// LivingEntity.pushEntities on the server: players whose boxes overlap push each other apart (Entity.push).
    fn push_entities(&mut self, i: usize) {
        for j in 0..self.players.len() {
            if j == i || self.players[j].dead || !intersects(&self.players[i].bb(), &self.players[j].bb()) {
                continue;
            }
            // players[j].push(players[i]).
            let (a, b) = (self.players[i].body.pos, self.players[j].body.pos);
            let mut d = a.x - b.x;
            let mut e = a.z - b.z;
            let mut f = jmax(d.abs(), e.abs());
            if f >= 0.01f32 as f64 {
                f = f.sqrt();
                d /= f;
                e /= f;
                let g = jmin(1.0 / f, 1.0);
                d *= g;
                e *= g;
                d *= 0.05f32 as f64;
                e *= 0.05f32 as f64;
                let pj = &mut self.players[j];
                pj.body.vel = pj.body.vel.add(V3::new(-d, 0.0, -e));
                pj.needs_sync = true;
                let pi = &mut self.players[i];
                pi.body.vel = pi.body.vel.add(V3::new(d, 0.0, e));
                pi.needs_sync = true;
            }
        }
    }

    /// ServerPlayer.doTick: Player.tick (LivingEntity.tick, then the attack timer, the pose), FoodData.tick, the
    /// health packet, the inventory sync.
    /// Entity.isInWall: a suffocating block's collider in the box 0.8 of the width wide and 1e-6 high around the
    /// eyes.
    fn in_wall(&self, i: usize) -> bool {
        let p = &self.players[i];
        let h = (0.6f32 * 0.8f32) as f64 / 2.0;
        let eye = V3::new(p.body.pos.x, p.body.pos.y + p.body.eye_height() as f64, p.body.pos.z);
        let (lo, hi) = ([eye.x - h, eye.y - 0.5e-6, eye.z - h], [eye.x + h, eye.y + 0.5e-6, eye.z + h]);
        for x in lo[0].floor() as i32..=hi[0].floor() as i32 {
            for y in lo[1].floor() as i32..=hi[1].floor() as i32 {
                for z in lo[2].floor() as i32..=hi[2].floor() as i32 {
                    let inside = |a: usize, c: i32| lo[a] < c as f64 + 1. && hi[a] > c as f64;
                    if self.level.solid(x, y, z) && inside(0, x) && inside(1, y) && inside(2, z) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn do_tick(&mut self, i: usize) {
        if self.players[i].dead {
            return;
        }
        let first_good = self.players[i].body.pos;
        self.players[i].first_good = first_good;
        // LivingEntity.baseTick: suffocating (isInWall), the hurt time.
        {
            if self.in_wall(i) {
                self.hurt(i, &DamageSource { kind: DamageKind::InWall, pos: None, attacker: None }, 1.0);
            }
            let p = &mut self.players[i];
            if p.hurt_time > 0 {
                p.hurt_time -= 1;
            }
        }
        // LivingEntity.tick: aiStep (the server's physics).
        {
            let level = &self.level;
            let p = &mut self.players[i];
            p.body.server_ai_step(level);
        }
        if self.players[i].dead {
            return;
        }
        self.push_entities(i);
        // Player.aiStep: updateSwingTime.
        {
            let p = &mut self.players[i];
            if p.swinging {
                p.swing_time += 1;
                if p.swing_time >= 6 {
                    p.swing_time = 0;
                    p.swinging = false;
                }
            } else {
                p.swing_time = 0;
            }
            p.head_yaw = p.body.yaw;
        }
        // Player.tick: the attack timer, a new item in hand restarts it, the cooldowns, the pose.
        {
            let level = &self.level;
            let p = &mut self.players[i];
            if p.current_impulse_context_reset_grace_time > 0 {
                p.current_impulse_context_reset_grace_time -= 1;
            }
            p.attack_strength_ticker += 1;
            let main = p.main_hand();
            if !(main.item == p.last_item_main.item && main.count == p.last_item_main.count && main.damage == p.last_item_main.damage) {
                if main.item != p.last_item_main.item {
                    p.reset_attack_strength();
                }
                p.last_item_main = main;
            }
            // updatePlayerPose (at the client's position: tickPlayer puts the player back after doTick).
            let at = first_good;
            let fits = |crouch: bool| {
                let mut b = p.body.bounding_box_at(at, crouch);
                for k in 0..3 {
                    b.min[k] += 1.0e-7;
                    b.max[k] -= 1.0e-7;
                }
                no_collision(level, &b)
            };
            let want = p.body.keys.shift;
            let pose = if fits(want) { want } else if fits(true) { true } else { p.body.crouch_pose };
            if pose != p.body.crouch_pose {
                p.body.crouch_pose = pose;
                p.pose_dirty = true;
            }
        }
        // FoodData.tick.
        self.food_tick(i);
        // tickPlayer: back to where the client said.
        self.players[i].body.pos = first_good;
        // The health packet.
        let p = &mut self.players[i];
        if p.health != p.last_sent_health || p.last_sent_food != p.food.level || (p.food.saturation == 0.0) != p.last_food_saturation_zero {
            let (hp, food, sat) = (p.health, p.food.level, p.food.saturation);
            p.last_sent_health = hp;
            p.last_sent_food = food;
            p.last_food_saturation_zero = sat == 0.0;
            self.send(i, S2C::Health { hp, food, sat });
        }
    }





    /// FoodData.tick.
    fn food_tick(&mut self, i: usize) {
        let p = &mut self.players[i];
        if p.food.exhaustion > 4.0 {
            p.food.exhaustion -= 4.0;
            if p.food.saturation > 0.0 {
                p.food.saturation = fmax(p.food.saturation - 1.0, 0.0);
            } else {
                p.food.level = (p.food.level - 1).max(0);
            }
        }
        if p.food.saturation > 0.0 && p.is_hurt() && p.food.level >= 20 {
            p.food.tick_timer += 1;
            if p.food.tick_timer >= 10 {
                let f = fmin(p.food.saturation, 6.0);
                p.heal(f / 6.0);
                p.food.add_exhaustion(f);
                p.food.tick_timer = 0;
            }
        } else if p.food.level >= 18 && p.is_hurt() {
            p.food.tick_timer += 1;
            if p.food.tick_timer >= 80 {
                p.heal(1.0);
                p.food.add_exhaustion(6.0);
                p.food.tick_timer = 0;
            }
        } else if p.food.level <= 0 {
            p.food.tick_timer += 1;
            if p.food.tick_timer >= 80 {
                p.food.tick_timer = 0;
                // Normal difficulty: starving stops at half a heart.
                if p.health > 1.0 {
                    self.hurt(i, &DamageSource { kind: DamageKind::Starve, pos: None, attacker: None }, 1.0);
                }
            }
        } else {
            p.food.tick_timer = 0;
        }
    }
}


/// LivingEntity.calculateFallDamage: floor((fall + 1e-6 - safe_fall_distance 3) * block multiplier 1 * fall_damage_multiplier 1)
fn fall_damage(fall: f64) -> i32 {
    ((fall + 1.0e-6 - 3.0) * 1.0f32 as f64 * 1.0).floor() as i32
}


/// CombatRules.getDamageAfterAbsorb: armor and toughness.
pub fn after_armor(f: f32, g: f32, h: f32) -> f32 {
    let i = 2.0f32 + h / 4.0f32;
    let j = clampf(g - f / i, g * 0.2f32, 20.0f32);
    let k = j / 25.0f32;
    let m = 1.0f32 - k;
    f * m
}

/// AABB.distanceToSqr(point).
pub fn bb_distance_sqr(b: &Bb, p: V3) -> f64 {
    let d = jmax(jmax(b.min[0] - p.x, p.x - b.max[0]), 0.0);
    let e = jmax(jmax(b.min[1] - p.y, p.y - b.max[1]), 0.0);
    let f = jmax(jmax(b.min[2] - p.z, p.z - b.max[2]), 0.0);
    d * d + e * e + f * f
}

/// ServerGamePacketListenerImpl.isEntityCollidingWithAnythingNew: a block in the new box (both shrunk by 1e-5)
/// that is not in the old one.
fn colliding_with_anything_new(w: &Level, old: &Bb, new: &Bb) -> bool {
    let e = 1.0e-5f32 as f64;
    let shrink = |b: &Bb| Bb { min: [b.min[0] + e, b.min[1] + e, b.min[2] + e], max: [b.max[0] - e, b.max[1] - e, b.max[2] - e] };
    let (old, new) = (shrink(old), shrink(new));
    for x in floor(new.min[0])..=floor(new.max[0]) {
        for y in floor(new.min[1])..=floor(new.max[1]) {
            for z in floor(new.min[2])..=floor(new.max[2]) {
                if w.solid(x, y, z) {
                    let c = Bb { min: [x as f64, y as f64, z as f64], max: [x as f64 + 1.0, y as f64 + 1.0, z as f64 + 1.0] };
                    if intersects(&new, &c) && !intersects(&old, &c) {
                        return true;
                    }
                }
            }
        }
    }
    false
}





/// Level.noCollision for block cubes.
pub fn no_collision(w: &Level, b: &Bb) -> bool {
    for x in floor(b.min[0])..=floor(b.max[0]) {
        for y in floor(b.min[1])..=floor(b.max[1]) {
            for z in floor(b.min[2])..=floor(b.max[2]) {
                if w.solid(x, y, z) {
                    let c = Bb { min: [x as f64, y as f64, z as f64], max: [x as f64 + 1.0, y as f64 + 1.0, z as f64 + 1.0] };
                    if intersects(b, &c) {
                        return false;
                    }
                }
            }
        }
    }
    true
}













