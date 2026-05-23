# =============================================================================
# tests/test_compliance.py
# Unit tests for AS1170.4 + Amendment 2 (2018) compliance calculations
# Run: python -m pytest tests/ -v
# =============================================================================

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from compliance import static_base_shear
from config import spectral_shape_De

# Reference parameters — Building 1 (Pre-1990), verified against
# manual calculation in report Section 4.1
B1_PARAMS = {
    'num_storeys':   2,
    'storey_height': 3.0,
    'num_bays':      3,
    'bay_width':     4.0,
    'floor_width':   8.0,
    'fc': 20.0, 'fy': 250.0,
    'dead_load': 5.0, 'live_load': 2.0,
    'Z': 0.11, 'mu': 2.0, 'Sp': 0.77, 'site_class': 'De',
}
W_TOTAL_B1 = 1075.2  # kN — verified


class TestSpectralShape:
    def test_short_period_plateau(self):
        """Ch capped at 2.35 for very short periods"""
        # Ch is now continuous; values near zero clamp to 2.35
        assert abs(spectral_shape_De(0.05) - 2.35) < 0.001

    def test_at_T1_eq_0p1(self):
        """At T=0.1s, descending branch gives Ch = 1.65"""
        # With the fix, the function uses min(2.35, 1.65*(0.1/T)^0.85)
        # At T=0.1 exactly: 1.65 * 1 = 1.65 (below the 2.35 cap)
        assert abs(spectral_shape_De(0.10) - 1.65) < 0.001

    def test_descending_branch(self):
        """Ch(0.288s) should be ~0.672 for Site De — Building 1 code period"""
        ch = spectral_shape_De(0.288)
        assert abs(ch - 0.672) < 0.005

    def test_long_period_monotone(self):
        """Ch(T>1.5) decreases with T"""
        ch_15 = spectral_shape_De(1.5)
        ch_2  = spectral_shape_De(2.0)
        ch_3  = spectral_shape_De(3.0)
        assert ch_15 > ch_2 > ch_3

    def test_continuity_at_T_0p1(self):
        """Ch is continuous across T = 0.1s (no jump).

        Previously had a 0.7 jump (2.35 → 1.65) at T=0.1. Now both sides
        evaluate the descending formula 1.65 × (0.1/T)^0.85 which has a
        natural slope of ~14 at T=0.1, so a 0.002 change in T produces
        an ~0.028 change in Ch. The test verifies there is no jump
        discontinuity, with tolerance set to the slope-implied bound.
        """
        # Test at T = 0.1 exactly from a very small offset
        left  = spectral_shape_De(0.0999)
        right = spectral_shape_De(0.1001)
        # Expected slope at T=0.1 is ~14, so 0.0002 change → 0.003 in Ch
        assert abs(left - right) < 0.01, \
            f"Discontinuity at T=0.1: left={left:.4f}, right={right:.4f}"

    def test_continuity_at_T_1p5(self):
        """Ch is continuous across T = 1.5s (no jump)"""
        left  = spectral_shape_De(1.499)
        right = spectral_shape_De(1.501)
        assert abs(left - right) < 0.005, \
            f"Discontinuity at T=1.5: left={left:.3f}, right={right:.3f}"


class TestAmendment2:
    """AS 1170.4 Amendment No. 2 (2018) Cl 3.3 minimum kpZ = 0.08."""

    def test_amendment_2_returned(self):
        """static_base_shear now returns kpZ as fourth element"""
        result = static_base_shear(B1_PARAMS, W_TOTAL_B1)
        assert len(result) == 4, "Expected (V, Ch, T1, kpZ)"

    def test_newcastle_unaffected(self):
        """Newcastle Z=0.11 > 0.08 so kpZ unchanged"""
        V, Ch, T1, kpZ = static_base_shear(B1_PARAMS, W_TOTAL_B1)
        assert abs(kpZ - 0.11) < 0.001

    def test_brisbane_minimum_applied(self):
        """Brisbane Z_raw=0.05 should raise to kpZ=0.08 (60% increase)"""
        brisbane = {**B1_PARAMS, 'Z': 0.05}
        V_bne, _, _, kpZ_bne = static_base_shear(brisbane, W_TOTAL_B1)
        assert abs(kpZ_bne - 0.08) < 0.001, \
            f"Brisbane kpZ should be 0.08, got {kpZ_bne}"

    def test_brisbane_60pct_higher_than_raw(self):
        """Brisbane base shear with Amdt 2 is 1.60x what raw Z would give"""
        V_raw = (0.05 / B1_PARAMS['mu']) * B1_PARAMS['Sp'] * 0.672 * W_TOTAL_B1
        V_amd2, _, _, _ = static_base_shear({**B1_PARAMS, 'Z': 0.05}, W_TOTAL_B1)
        # Amdt 2 multiplies by 0.08/0.05 = 1.6
        assert V_amd2 / V_raw > 1.5


class TestStaticBaseShear:
    def test_building1_base_shear(self):
        """V = 30.6 kN for Building 1 — verified manually"""
        V, Ch, T1, kpZ = static_base_shear(B1_PARAMS, W_TOTAL_B1)
        assert abs(V - 30.6) < 0.5, f"Expected ~30.6 kN, got {V:.1f} kN"

    def test_building1_period(self):
        """T1 = 0.075 * 6^0.75 = 0.288 s"""
        V, Ch, T1, kpZ = static_base_shear(B1_PARAMS, W_TOTAL_B1)
        assert abs(T1 - 0.288) < 0.001

    def test_building1_vw_ratio(self):
        """V/W = 0.0285 for Building 1"""
        V, Ch, T1, kpZ = static_base_shear(B1_PARAMS, W_TOTAL_B1)
        ratio = V / W_TOTAL_B1
        assert abs(ratio - 0.0285) < 0.001

    def test_higher_mu_lower_shear(self):
        """Higher ductility -> lower design base shear"""
        params2 = {**B1_PARAMS, 'mu': 3.0, 'Sp': 0.67}
        params3 = {**B1_PARAMS, 'mu': 4.0, 'Sp': 0.67}
        V1, *_ = static_base_shear(B1_PARAMS, W_TOTAL_B1)
        V2, *_ = static_base_shear(params2, W_TOTAL_B1)
        V3, *_ = static_base_shear(params3, W_TOTAL_B1)
        assert V1 > V2 > V3

    def test_minimum_base_shear(self):
        """V >= 0.01 * W even for long period structures"""
        long_period_params = {**B1_PARAMS, 'num_storeys': 4}
        W = W_TOTAL_B1 * 2
        V, *_ = static_base_shear(long_period_params, W)
        assert V >= 0.01 * W

    def test_building2_shear(self):
        """Building 2 (mu=3.0): V/W ~0.0174"""
        params2 = {**B1_PARAMS, 'mu': 3.0, 'Sp': 0.67}
        V, *_ = static_base_shear(params2, W_TOTAL_B1)
        ratio = V / W_TOTAL_B1
        assert abs(ratio - 0.0174) < 0.002

    def test_building3_shear(self):
        """Building 3 (mu=4.0): V/W ~0.0130"""
        params3 = {**B1_PARAMS, 'mu': 4.0, 'Sp': 0.67}
        V, *_ = static_base_shear(params3, W_TOTAL_B1)
        ratio = V / W_TOTAL_B1
        assert abs(ratio - 0.0130) < 0.002


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-v'])
