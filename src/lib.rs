// SPDX-License-Identifier: AGPL-3.0-only
// sword-sim by jo29_29. Licensed under the GNU AGPL v3 only with additional terms: see LICENSE and ADDITIONAL_TERMS.
//! A headless, tick-exact Minecraft 1.21.11 sword-duel simulator (server, client prediction and network lag), with a
//! C ABI for batched reinforcement-learning environments (`env`).
mod env;
mod fight;
mod sim;
