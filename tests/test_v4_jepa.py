"""Unit test suite for V4 Pokemon JEPA World Models.

Tests each gap identified in pokemon-jepa-flaws.md:
- 1.1 True Latent JEPA: EMA target encoder, stop-gradient, VICReg loss
- 1.2 Action Conditioning: Action-conditioned latent transition sensitivity
- 1.3 Non-Circular Probe evaluation
- 1.4 Persistence Baseline & R^2 over persistence calculation
- 2.1 Positional, Slot, Side, and Active token embeddings
- 2.2 Canonical slot ordering (no array swapping on switches)
- 2.3 Split-first vocabulary fitting with <unk> mapping
- 3.1 Stat stage boosts and hazard/weather parsing
- 3.2 Masked species head and Teambuilder ranking
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from v4_parser import (
    BattleState,
    PokemonVocab,
    PokemonState,
    SideState,
    find_or_assign_slot,
    parse_v4_replay,
    make_v4_transitions,
    snapshot_board,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
)
from v4_models import (
    PositionalTransformerEncoder,
    ActionConditionedPredictor,
    TrueJEPA,
    Teambuilder,
)
from v4_baselines import compute_hp_metrics, compute_calibration_metrics


SAMPLE_REPLAY_LOG = """|j|☆Player 1
|j|☆Player 2
|gametype|singles
|player|p1|Player 1|1|1200
|player|p2|Player 2|2|1200
|teamsize|p1|6
|teamsize|p2|6
|gen|9
|tier|[Gen 9] PU
|clearpoke
|poke|p1|Delphox, M|
|poke|p1|Florges, F|
|poke|p1|Mudsdale, M|
|poke|p1|Skuntank, F|
|poke|p1|Rotom-Mow|
|poke|p1|Milotic, F|
|poke|p2|Decidueye-Hisui, F|
|poke|p2|Articuno-Galar|
|poke|p2|Bellibolt, F|
|poke|p2|Tauros-Paldea-Blaze, M|
|poke|p2|Hitmonlee, M|
|poke|p2|Appletun, F|
|teampreview
|start
|switch|p1a: Rotom|Rotom-Mow|100/100
|switch|p2a: Bellibolt|Bellibolt, F|100/100
|turn|1
|switch|p2a: Appletun|Appletun, F|100/100
|move|p1a: Rotom|Trick|p2a: Appletun
|-activate|p1a: Rotom|move: Trick|[of] p2a: Appletun
|turn|2
|-sidestart|p2: Player 2|move: Stealth Rock
|-weather|SunnyDay
|-boost|p1a: Rotom|spa|2
|move|p1a: Rotom|Volt Switch|p2a: Appletun
|-resisted|p2a: Appletun
|-damage|p2a: Appletun|80/100
|switch|p1a: Florges|Florges, F|100/100|[from] Volt Switch
|move|p2a: Appletun|Apple Acid|p1a: Florges
|-damage|p1a: Florges|75/100
|turn|3
|move|p1a: Florges|Moonblast|p2a: Appletun
|-damage|p2a: Appletun|20/100
|move|p2a: Appletun|Recover|p2a: Appletun
|-heal|p2a: Appletun|70/100
|turn|4
|move|p1a: Florges|Moonblast|p2a: Appletun
|-damage|p2a: Appletun|0 fnt
|faint|p2a: Appletun
|win|Player 1
"""


def test_canonical_slot_ordering_and_no_swapping():
    """Flaw 2.2: Ensure slots 0..5 are fixed and never swapped upon switches."""
    vocab = PokemonVocab()
    parsed = parse_v4_replay(SAMPLE_REPLAY_LOG, vocab, update_vocab=True)
    assert len(parsed) >= 3

    # Turn 1: Rotom (slot 4) active for P1
    t1 = parsed[0]
    assert t1["p1_active_idx"] == 4
    # Slot 0 must remain Delphox and Slot 4 must remain Rotom-Mow
    delphox_id = vocab.get_species("delphox")
    rotom_id = vocab.get_species("rotom-mow")
    florges_id = vocab.get_species("florges")

    assert int(t1["slots"][0][0]) == delphox_id
    assert int(t1["slots"][4][0]) == rotom_id
    assert int(t1["slots"][1][0]) == florges_id

    # Turn 3: Switched to Florges (slot 1) during turn 2
    t3 = parsed[2]
    assert t3["p1_active_idx"] == 1
    # Check that slot positions did NOT permute!
    assert int(t3["slots"][0][0]) == delphox_id
    assert int(t3["slots"][4][0]) == rotom_id
    assert int(t3["slots"][1][0]) == florges_id
    # Active indicator must correctly show slot 1 active and slot 4 bench
    assert t3["slots"][1][3] == 1.0  # is_active
    assert t3["slots"][4][3] == 0.0  # is_active


def test_tactical_boosts_hazards_weather_parsing():
    """Flaw 3.1: Ensure boosts, hazards, and weather are correctly parsed into state."""
    vocab = PokemonVocab()
    parsed = parse_v4_replay(SAMPLE_REPLAY_LOG, vocab, update_vocab=True)

    # Set during turn 2, visible at start of turn 3: Stealth Rock on P2 side, SunnyDay active
    t3 = parsed[2]
    # Weather: Sun is ID 1
    assert t3["field"][0] == 1.0  # weather = sun
    # P2 Stealth rock hazard: field index 5
    assert t3["field"][5] == 1.0  # p2 stealth rock = 1


def test_action_conditioning_parsing():
    """Flaw 1.2: Ensure actions (move vs switch, targets) are parsed."""
    vocab = PokemonVocab()
    parsed = parse_v4_replay(SAMPLE_REPLAY_LOG, vocab, update_vocab=True)

    t1 = parsed[0]
    # P1 used move (Trick), P2 switched to Appletun (slot 5)
    assert t1["p1_action"][0] == 1  # move
    assert t1["p2_action"][0] == 2  # switch
    assert t1["p2_action"][1] == 5  # target slot 5


def test_split_first_vocab_isolation():
    """Flaw 2.3: Unseen test species must map cleanly to <unk> without leaking."""
    train_vocab = PokemonVocab()
    train_vocab.add_species("delphox")
    train_vocab.add_species("florges")

    assert train_vocab.get_species("delphox") > 2
    assert train_vocab.get_species("unknownmon") == UNK_TOKEN_ID


def test_positional_transformer_breaks_permutation_equivariance():
    """Flaw 2.1: Positional/side/slot embeddings ensure attention knows board structure."""
    encoder = PositionalTransformerEncoder(species_vocab_size=50, d_model=32, nhead=2, num_layers=1, latent_dim=32)
    encoder.eval()

    # Create batch with 2 slots having identical features but different slot indices (0 vs 1)
    slots = torch.zeros((1, 12, 9))
    slots[:, :, 0] = 5.0  # all same species
    slots[:, :, 1] = 1.0  # hp
    field = torch.zeros((1, 9))

    out1 = encoder(slots, field)["slot_tokens"]
    # Token 0 (slot 0) and Token 1 (slot 1) MUST have distinct representations because of slot_embed!
    diff = (out1[:, 0, :] - out1[:, 1, :]).norm().item()
    assert diff > 1e-3, "Tokens at different slots must differ due to positional slot embeddings!"


def test_action_conditioned_predictor_sensitivity():
    """Flaw 1.2: ActionConditionedPredictor produces distinct predictions for different actions."""
    predictor = ActionConditionedPredictor(latent_dim=32, move_vocab_size=50, action_dim=32)
    predictor.eval()

    context = torch.randn(2, 32)
    act_move = torch.tensor([[1, 10, 0], [1, 10, 0]])     # move 10
    act_switch = torch.tensor([[2, 3, 0], [2, 3, 0]])     # switch to slot 3
    act_p2 = torch.tensor([[1, 5, 0], [1, 5, 0]])

    pred1 = predictor(context, act_move, act_p2)
    pred2 = predictor(context, act_switch, act_p2)

    diff = (pred1 - pred2).norm(dim=-1)
    assert (diff > 1e-3).all(), "Predictor must produce distinct latents for attack vs switch!"


def test_true_jepa_target_encoder_ema_and_vicreg():
    """Flaw 1.1: True JEPA has EMA target encoder, stop-gradient, and VICReg loss."""
    jepa = TrueJEPA(species_vocab_size=50, move_vocab_size=50, d_model=32, nhead=2, latent_dim=32, ema_decay=0.99)
    jepa.train()

    # Verify target encoder has requires_grad == False
    for p in jepa.target_encoder.parameters():
        assert not p.requires_grad, "Target encoder must have stop-gradient!"

    # Forward pass
    b = 4
    curr_slots = torch.randn(b, 12, 9).abs()
    curr_field = torch.zeros(b, 9)
    nxt_slots = torch.randn(b, 12, 9).abs()
    nxt_field = torch.zeros(b, 9)
    p1_act = torch.tensor([[1, 2, 0]] * b)
    p2_act = torch.tensor([[1, 3, 0]] * b)

    loss, metrics = jepa(curr_slots, curr_field, nxt_slots, nxt_field, p1_act, p2_act)
    assert loss.item() > 0.0
    assert "inv_loss" in metrics
    assert "var_loss" in metrics
    assert "cov_loss" in metrics

    # Test EMA update
    initial_online_weight = next(jepa.online_encoder.parameters()).clone()
    initial_target_weight = next(jepa.target_encoder.parameters()).clone()

    loss.backward()
    opt = torch.optim.SGD(jepa.online_encoder.parameters(), lr=0.1)
    opt.step()

    jepa.update_target_encoder()
    updated_target_weight = next(jepa.target_encoder.parameters())
    # Target weight should have changed slightly towards new online weight
    assert not torch.equal(initial_target_weight, updated_target_weight)


def test_persistence_baseline_and_r2_metric():
    """Flaw 1.4: Strict persistence baseline and R^2 over persistence computation."""
    y_curr = np.array([[1.0, 1.0, 0.5, 0.0], [1.0, 0.8, 0.0, 1.0]])
    y_true = np.array([[0.8, 1.0, 0.5, 0.0], [1.0, 0.4, 0.0, 1.0]])  # slot 0 and slot 1 took damage

    # Model predicting perfect target
    perfect_pred = y_true.copy()
    m_perfect = compute_hp_metrics(y_true, perfect_pred, y_curr)
    assert m_perfect["hp_mse"] == 0.0
    assert m_perfect["r2_over_persistence"] == 1.0
    assert m_perfect["persist_mse"] > 0.0

    # Model predicting naive persistence
    persist_pred = y_curr.copy()
    m_persist = compute_hp_metrics(y_true, persist_pred, y_curr)
    assert m_persist["r2_over_persistence"] == 0.0


def test_teambuilder_masked_species_completion():
    """Flaw 3.2: Teambuilder masks 6th slot and suggests species with valid probability distribution."""
    vocab = PokemonVocab()
    for s in ["delphox", "florges", "mudsdale", "skuntank", "rotommow", "milotic", "bellibolt", "appletun"]:
        vocab.add_species(s)

    encoder = PositionalTransformerEncoder(len(vocab.species), d_model=32, nhead=2, latent_dim=32)
    device = torch.device("cpu")
    tb = Teambuilder(encoder, vocab, device)

    suggestions = tb.suggest_sixth(["delphox", "florges", "mudsdale", "skuntank", "rotommow"], top_k=3)
    assert len(suggestions) <= 3
    # Verify no already selected species is suggested
    suggested_names = [s[0] for s in suggestions]
    for s in ["delphox", "florges", "mudsdale", "skuntank", "rotommow"]:
        assert s not in suggested_names, f"Species {s} already in team should not be suggested!"
