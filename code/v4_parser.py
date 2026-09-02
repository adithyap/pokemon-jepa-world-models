"""Pokemon Showdown replay parser for V4 JEPA World Models study.

Addresses the following flaws identified in pokemon-jepa-flaws.md:
- 2.2 Canonical Slot Ordering: Slots 0..5 for P1 and 6..11 for P2 are assigned once and never swapped.
- 2.1 Positional & Slot Metadata: Preserves exact slot_id, side_id, and active status.
- 1.2 Action Conditioning: Parses both players' actions (move vs switch, target, tera) per turn.
- 2.3 Split-First Vocabulary: Vocabulary is fitted strictly on training data with <pad>, <unk>, and <mask_token> tokens.
- 3.1 Tactical State Parsing: Parses stat boosts (-6..+6), hazards, weather, and status.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


STATUS_TO_ID = {"": 0, "par": 1, "psn": 2, "tox": 3, "brn": 4, "frz": 5, "slp": 6}
WEATHER_TO_ID = {"none": 0, "sun": 1, "rain": 2, "sand": 3, "snow": 4}
BOOST_STATS = ("atk", "def", "spa", "spd", "spe")

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
MASK_TOKEN_ID = 2


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def hp_fraction(detail: str) -> float:
    token = detail.split()[0]
    if token == "0" or "fnt" in detail:
        return 0.0
    if "/" not in token:
        return 1.0
    left, right = token.split("/", 1)
    try:
        denom = float(right.rstrip("%"))
        return max(0.0, min(1.0, float(left) / denom)) if denom > 0 else 1.0
    except ValueError:
        return 1.0


@dataclass
class PokemonState:
    species: str = "none"
    hp: float = 1.0
    status: str = ""
    boosts: dict[str, int] = field(default_factory=lambda: {s: 0 for s in BOOST_STATS})

    def copy(self) -> PokemonState:
        return PokemonState(
            species=self.species,
            hp=self.hp,
            status=self.status,
            boosts=dict(self.boosts),
        )

    def reset_boosts(self) -> None:
        for s in BOOST_STATS:
            self.boosts[s] = 0


@dataclass
class SideState:
    team: list[PokemonState] = field(default_factory=lambda: [PokemonState() for _ in range(6)])
    active_idx: int = 0
    hazards: dict[str, int] = field(
        default_factory=lambda: {
            "stealth_rock": 0,
            "spikes": 0,
            "toxic_spikes": 0,
            "sticky_web": 0,
        }
    )

    def copy(self) -> SideState:
        res = SideState()
        res.team = [p.copy() for p in self.team]
        res.active_idx = self.active_idx
        res.hazards = dict(self.hazards)
        return res


@dataclass
class BattleState:
    turn: int = 0
    weather: str = "none"
    p1: SideState = field(default_factory=SideState)
    p2: SideState = field(default_factory=SideState)

    def copy(self) -> BattleState:
        res = BattleState(turn=self.turn, weather=self.weather)
        res.p1 = self.p1.copy()
        res.p2 = self.p2.copy()
        return res


class PokemonVocab:
    """Vocabulary mapping for species and moves, fitted strictly on training data."""

    def __init__(self) -> None:
        self.species: dict[str, int] = {
            "<pad>": PAD_TOKEN_ID,
            "<unk>": UNK_TOKEN_ID,
            "<mask_token>": MASK_TOKEN_ID,
            "none": PAD_TOKEN_ID,
        }
        self.moves: dict[str, int] = {
            "<none>": 0,
            "<unk>": 1,
        }

    def add_species(self, name: str) -> int:
        norm = normalized_name(name)
        if not norm:
            return PAD_TOKEN_ID
        if norm not in self.species:
            self.species[norm] = len(self.species)
        return self.species[norm]

    def add_move(self, name: str) -> int:
        norm = normalized_name(name)
        if not norm:
            return 0
        if norm not in self.moves:
            self.moves[norm] = len(self.moves)
        return self.moves[norm]

    def get_species(self, name: str) -> int:
        norm = normalized_name(name)
        return self.species.get(norm, UNK_TOKEN_ID)

    def get_move(self, name: str) -> int:
        norm = normalized_name(name)
        return self.moves.get(norm, 1)


@dataclass
class TurnAction:
    """Player action taken on a turn."""
    # kind: 0 = none / pass / unknown, 1 = move, 2 = switch
    kind: int = 0
    # target: move_id if kind==1, or switch slot (0..5) if kind==2
    target: int = 0
    tera: int = 0


def snapshot_slot_features(
    p: PokemonState, is_active: bool, vocab: PokemonVocab
) -> list[float]:
    """Encode a single pokemon slot into numerical features:
    [species_id, hp, status_id, is_active, atk_boost, def_boost, spa_boost, spd_boost, spe_boost]
    """
    spec_id = float(vocab.get_species(p.species))
    hp_val = float(p.hp)
    stat_id = float(STATUS_TO_ID.get(p.status, 0))
    active_val = 1.0 if is_active else 0.0
    boosts = [float(p.boosts.get(s, 0)) / 6.0 for s in BOOST_STATS]
    return [spec_id, hp_val, stat_id, active_val] + boosts


def snapshot_board(
    state: BattleState, vocab: PokemonVocab
) -> dict[str, Any]:
    """Snapshot 12 slots canonically with positional and tactical indicators."""
    slots: list[list[float]] = []
    # Slots 0..5: P1
    for idx, p in enumerate(state.p1.team):
        is_active = (idx == state.p1.active_idx)
        slots.append(snapshot_slot_features(p, is_active, vocab))
    # Slots 6..11: P2
    for idx, p in enumerate(state.p2.team):
        is_active = (idx == state.p2.active_idx)
        slots.append(snapshot_slot_features(p, is_active, vocab))

    field_features = [
        float(WEATHER_TO_ID.get(state.weather, 0)),
        float(state.p1.hazards["stealth_rock"]),
        float(state.p1.hazards["spikes"]) / 3.0,
        float(state.p1.hazards["toxic_spikes"]) / 2.0,
        float(state.p1.hazards["sticky_web"]),
        float(state.p2.hazards["stealth_rock"]),
        float(state.p2.hazards["spikes"]) / 3.0,
        float(state.p2.hazards["toxic_spikes"]) / 2.0,
        float(state.p2.hazards["sticky_web"]),
    ]

    hps = [p.hp for p in state.p1.team] + [p.hp for p in state.p2.team]
    return {
        "turn": state.turn,
        "slots": slots,
        "field": field_features,
        "hps": hps,
        "p1_active_idx": state.p1.active_idx,
        "p2_active_idx": state.p2.active_idx,
    }


def find_or_assign_slot(side: SideState, species: str) -> int:
    """Find slot for species without swapping! Assigns to first empty slot if unassigned."""
    norm = normalized_name(species)
    # Search existing slots
    for idx, p in enumerate(side.team):
        if p.species != "none" and normalized_name(p.species) == norm:
            return idx
    # Otherwise place in first empty slot
    for idx, p in enumerate(side.team):
        if p.species == "none":
            p.species = species
            return idx
    return 0


def parse_v4_replay(
    log: str,
    vocab: PokemonVocab,
    update_vocab: bool = False,
) -> list[dict[str, Any]]:
    """Parse a Showdown replay into turn-by-turn transitions with:
    - Fixed canonical slot positions
    - Action conditioning for both players
    - Stat boosts, hazards, weather
    """
    state = BattleState()
    player_to_side: dict[str, str] = {}
    p1_count = 0
    p2_count = 0

    lines = log.splitlines()

    # Pass 1: Parse team preview and players
    for line in lines:
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        event = parts[1]

        if event == "player" and len(parts) >= 4:
            player_to_side[parts[3].strip()] = parts[2].strip()

        elif event == "poke" and len(parts) >= 4:
            side = parts[2][:2]
            species = parts[3].split(",")[0].strip()
            if update_vocab:
                vocab.add_species(species)
            if side == "p1" and p1_count < 6:
                state.p1.team[p1_count].species = species
                p1_count += 1
            elif side == "p2" and p2_count < 6:
                state.p2.team[p2_count].species = species
                p2_count += 1

    # Pass 2: Step through battle line by line, capturing turns and actions
    turn_rows: list[dict[str, Any]] = []
    current_turn_data: dict[str, Any] | None = None

    p1_act = TurnAction()
    p2_act = TurnAction()
    winner_side: str | None = None

    for line in lines:
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        event = parts[1]

        if event == "turn" and len(parts) >= 3:
            turn_num = int(parts[2])
            if current_turn_data is not None:
                current_turn_data["p1_action"] = [p1_act.kind, p1_act.target, p1_act.tera]
                current_turn_data["p2_action"] = [p2_act.kind, p2_act.target, p2_act.tera]
                turn_rows.append(current_turn_data)

            p1_act = TurnAction()
            p2_act = TurnAction()

            state.turn = turn_num
            current_turn_data = snapshot_board(state, vocab)

        elif event == "switch" or event == "drag":
            if len(parts) >= 4:
                side_str = parts[2][:2]
                side = state.p1 if side_str == "p1" else state.p2
                species = parts[3].split(",")[0].strip()
                if update_vocab:
                    vocab.add_species(species)

                slot_idx = find_or_assign_slot(side, species)
                side.team[side.active_idx].reset_boosts()
                side.active_idx = slot_idx
                side.team[slot_idx].reset_boosts()
                if len(parts) >= 5:
                    side.team[slot_idx].hp = hp_fraction(parts[4])

                is_subturn = any("from" in p for p in parts[4:]) if len(parts) > 4 else False
                act = p1_act if side_str == "p1" else p2_act
                if act.kind == 0 and not is_subturn:
                    act.kind = 2
                    act.target = slot_idx

        elif event == "move" and len(parts) >= 4:
            side_str = parts[2][:2]
            move_name = parts[3].strip()
            if update_vocab:
                move_id = vocab.add_move(move_name)
            else:
                move_id = vocab.get_move(move_name)

            act = p1_act if side_str == "p1" else p2_act
            if act.kind == 0:
                act.kind = 1
                act.target = move_id

        elif event == "-terastallize" and len(parts) >= 4:
            side_str = parts[2][:2]
            act = p1_act if side_str == "p1" else p2_act
            act.tera = 1

        elif event in ("-damage", "-heal") and len(parts) >= 4:
            side_str = parts[2][:2]
            side = state.p1 if side_str == "p1" else state.p2
            side.team[side.active_idx].hp = hp_fraction(parts[3])

        elif event == "faint" and len(parts) >= 3:
            side_str = parts[2][:2]
            side = state.p1 if side_str == "p1" else state.p2
            side.team[side.active_idx].hp = 0.0
            side.team[side.active_idx].reset_boosts()

        elif event == "-boost" and len(parts) >= 5:
            side_str = parts[2][:2]
            side = state.p1 if side_str == "p1" else state.p2
            stat = parts[3].strip()
            amount = int(parts[4]) if parts[4].isdigit() else 1
            if stat in side.team[side.active_idx].boosts:
                side.team[side.active_idx].boosts[stat] = min(
                    6, side.team[side.active_idx].boosts[stat] + amount
                )

        elif event == "-unboost" and len(parts) >= 5:
            side_str = parts[2][:2]
            side = state.p1 if side_str == "p1" else state.p2
            stat = parts[3].strip()
            amount = int(parts[4]) if parts[4].isdigit() else 1
            if stat in side.team[side.active_idx].boosts:
                side.team[side.active_idx].boosts[stat] = max(
                    -6, side.team[side.active_idx].boosts[stat] - amount
                )

        elif event in ("-clearboost", "-clearallboost"):
            state.p1.team[state.p1.active_idx].reset_boosts()
            state.p2.team[state.p2.active_idx].reset_boosts()

        elif event == "-status" and len(parts) >= 4:
            side_str = parts[2][:2]
            side = state.p1 if side_str == "p1" else state.p2
            side.team[side.active_idx].status = parts[3].strip()

        elif event == "-curestatus" and len(parts) >= 3:
            side_str = parts[2][:2]
            side = state.p1 if side_str == "p1" else state.p2
            side.team[side.active_idx].status = ""

        elif event == "-sidestart" and len(parts) >= 4:
            side_str = parts[2][:2]
            side = state.p1 if side_str == "p1" else state.p2
            effect = parts[3].lower()
            if "stealth rock" in effect:
                side.hazards["stealth_rock"] = 1
            elif "spikes" in effect and "toxic" not in effect:
                side.hazards["spikes"] = min(3, side.hazards["spikes"] + 1)
            elif "toxic spikes" in effect:
                side.hazards["toxic_spikes"] = min(2, side.hazards["toxic_spikes"] + 1)
            elif "sticky web" in effect:
                side.hazards["sticky_web"] = 1

        elif event == "-sideend" and len(parts) >= 4:
            side_str = parts[2][:2]
            side = state.p1 if side_str == "p1" else state.p2
            effect = parts[3].lower()
            if "stealth rock" in effect:
                side.hazards["stealth_rock"] = 0
            elif "spikes" in effect and "toxic" not in effect:
                side.hazards["spikes"] = 0
            elif "toxic spikes" in effect:
                side.hazards["toxic_spikes"] = 0
            elif "sticky web" in effect:
                side.hazards["sticky_web"] = 0

        elif event == "-weather" and len(parts) >= 3:
            w_str = parts[2].lower()
            if "sun" in w_str:
                state.weather = "sun"
            elif "rain" in w_str:
                state.weather = "rain"
            elif "sand" in w_str:
                state.weather = "sand"
            elif "snow" in w_str or "hail" in w_str:
                state.weather = "snow"
            else:
                state.weather = "none"

        elif event == "win" and len(parts) >= 3:
            winner_name = parts[2].strip()
            winner_side = player_to_side.get(winner_name)

    if current_turn_data is not None:
        current_turn_data["p1_action"] = [p1_act.kind, p1_act.target, p1_act.tera]
        current_turn_data["p2_action"] = [p2_act.kind, p2_act.target, p2_act.tera]
        turn_rows.append(current_turn_data)

    if winner_side is not None:
        win_p1 = 1.0 if winner_side == "p1" else 0.0
        for row in turn_rows:
            row["winner_p1"] = win_p1

    return [r for r in turn_rows if "winner_p1" in r]


def make_v4_transitions(replays_parsed: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Convert turn rows to transition examples (s_t, a_t, s_{t+1}, delta_hp)."""
    transitions: list[dict[str, Any]] = []
    for replay in replays_parsed:
        for idx in range(len(replay) - 1):
            curr = replay[idx]
            nxt = replay[idx + 1]
            hps_curr = curr["hps"]
            hps_nxt = nxt["hps"]
            delta_hp = [float(h_n - h_c) for h_c, h_n in zip(hps_curr, hps_nxt)]
            transitions.append({
                "curr": curr,
                "next": nxt,
                "p1_action": curr["p1_action"],
                "p2_action": curr["p2_action"],
                "delta_hp": delta_hp,
                "winner_p1": curr["winner_p1"],
            })
    return transitions
