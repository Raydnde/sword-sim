// SPDX-License-Identifier: AGPL-3.0-only
// sword-sim by jo29_29. Licensed under the GNU AGPL v3 only with additional terms: see LICENSE and ADDITIONAL_TERMS.
//! The game: Minecraft 1.21.11's rules for a sword duel, tick for tick.
//! - `world`: blocks, collision, movement physics, items and enchantments.
//! - `server`: the dedicated server (players, combat, damage, knockback, packets).
//! - `client`: the client's own prediction and what it sends, and `Duel`, which joins two clients to a server over
//!   a lagged connection.
pub mod client;
pub mod server;
pub mod world;
