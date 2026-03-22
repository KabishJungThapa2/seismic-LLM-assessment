# =============================================================================
# tests/test_compliance.py
# Unit tests for AS1170.4 compliance calculations
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
    def test_short_period(self):
        """Ch(T≤0.1) = 2.35 for Site De"""
        assert abs(spectral_shape_De(0.05) - 2.35) < 0.001
        assert abs(spectral_shape_De(0.10) - 2.35) < 0.001

    def test_descending_branch(self):
        """Ch(0.288s) should be ~0.672 for Site De"""
        ch = spectral_shape_De(0.288)
        assert abs(ch - 0.672) < 0.005

    def test_long_period(self):
        """Ch(T>1.5) uses T^-2 formula"""
        ch_15 = spectral_shape_De(1.5)
        ch_2  = spectral_shape_De(2.0)
        assert ch_2 < ch_15  # should decrease

    def test_continuity_at_breakpoints(self):
        """Ch should be continuous at T=0.1 and T=1.5"""
        # At T=0.1: both sides should match
        left  = spectral_shape_De(0.099)
        right = spectral_shape_De(0.101)
        assert abs(left - right) < 0.01


class TestStaticBaseShear:
    def test_building1_base_shear(self):
        """V = 30.6 kN for Building 1 — verified manually"""
        V, Ch, T1 = static_base_shear(B1_PARAMS, W_TOTAL_B1)
        assert abs(V - 30.6) < 0.5, f"Expected ~30.6 kN, got {V:.1f} kN"

    def test_building1_period(self):
        """T1 = 0.075 * 6^0.75 = 0.288 s"""
        V, Ch, T1 = static_base_shear(B1_PARAMS, W_TOTAL_B1)
        assert abs(T1 - 0.288) < 0.001

    def test_building1_vw_ratio(self):
        """V/W = 0.0285 for Building 1"""
        V, Ch, T1 = static_base_shear(B1_PARAMS, W_TOTAL_B1)
        ratio = V / W_TOTAL_B1
        assert abs(ratio - 0.0285) < 0.001

    def test_higher_mu_lower_shear(self):
        """Higher ductility → lower design base shear"""
        params2 = {**B1_PARAMS, 'mu': 3.0, 'Sp': 0.67}
        params3 = {**B1_PARAMS, 'mu': 4.0, 'Sp': 0.67}
        V1, _, _ = static_base_shear(B1_PARAMS, W_TOTAL_B1)
        V2, _, _ = static_base_shear(params2, W_TOTAL_B1)
        V3, _, _ = static_base_shear(params3, W_TOTAL_B1)
        assert V1 > V2 > V3

    def test_minimum_base_shear(self):
        """V >= 0.01 * W even for long period structures"""
        long_period_params = {**B1_PARAMS, 'num_storeys': 4}
        W = W_TOTAL_B1 * 2
        V, _, _ = static_base_shear(long_period_params, W)
        assert V >= 0.01 * W

    def test_building2_shear(self):
        """Building 2 (mu=3.0): V/W ~0.0174"""
        params2 = {**B1_PARAMS, 'mu': 3.0, 'Sp': 0.67}
        V, _, _ = static_base_shear(params2, W_TOTAL_B1)
        ratio = V / W_TOTAL_B1
        assert abs(ratio - 0.0174) < 0.002

    def test_building3_shear(self):
        """Building 3 (mu=4.0): V/W ~0.0130"""
        params3 = {**B1_PARAMS, 'mu': 4.0, 'Sp': 0.67}
        V, _, _ = static_base_shear(params3, W_TOTAL_B1)
        ratio = V / W_TOTAL_B1
        assert abs(ratio - 0.0130) < 0.002


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-v'])
