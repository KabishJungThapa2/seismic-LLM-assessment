# =============================================================================
# tests/test_extractor.py
# Unit tests for the parameter extraction module
# Run: python -m pytest tests/ -v
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from extractor import _demo_extract, validate, _detect_era, _detect_storeys, _detect_geometry, _detect_zone


class TestEraDetection:
    def test_pre1990_by_year(self):
        assumptions = []
        assert _detect_era("built in 1975", assumptions) == "pre-1990"

    def test_pre1990_by_keyword(self):
        assumptions = []
        assert _detect_era("old heritage home with brick veneer", assumptions) == "pre-1990"

    def test_post1990_by_year(self):
        assumptions = []
        assert _detect_era("built in 1998 after the Newcastle earthquake", assumptions) == "post-1990"

    def test_post2010_by_year(self):
        assumptions = []
        assert _detect_era("modern home built in 2018", assumptions) == "post-2010"

    def test_post2010_by_keyword(self):
        assumptions = []
        assert _detect_era("recently built fully ductile building", assumptions) == "post-2010"

    def test_unknown_defaults_pre1990(self):
        assumptions = []
        era = _detect_era("a house in australia", assumptions)
        assert era == "pre-1990"
        assert len(assumptions) > 0  # should flag assumption


class TestStoreyDetection:
    def test_two_storey_text(self):
        assumptions = []
        assert _detect_storeys("a two storey residential building", assumptions) == 2

    def test_two_storey_number(self):
        assumptions = []
        assert _detect_storeys("a 2 storey home", assumptions) == 2

    def test_single_storey(self):
        assumptions = []
        assert _detect_storeys("a single storey bungalow", assumptions) == 1

    def test_three_storey(self):
        assumptions = []
        assert _detect_storeys("a 3-storey apartment block", assumptions) == 3

    def test_defaults_to_2(self):
        assumptions = []
        n = _detect_storeys("a house in newcastle", assumptions)
        assert n == 2


class TestGeometryDetection:
    def test_12x8_pattern(self):
        assumptions = []
        length, width, bays, bay_w = _detect_geometry(
            "floor plan 12m x 8m", assumptions)
        assert length == 12.0
        assert width == 8.0
        assert bays == 3

    def test_by_pattern(self):
        assumptions = []
        length, width, bays, bay_w = _detect_geometry(
            "12 metres by 8 metres", assumptions)
        assert length == 12.0

    def test_default_when_missing(self):
        assumptions = []
        length, width, bays, bay_w = _detect_geometry(
            "a house with no dimensions given", assumptions)
        assert length == 12.0
        assert width == 8.0
        assert bays == 3
        assert len(assumptions) > 0

    def test_width_exceeds_length_swapped(self):
        """Should always ensure length >= width"""
        assumptions = []
        length, width, bays, bay_w = _detect_geometry(
            "8m x 12m floor plan", assumptions)
        assert length >= width


class TestZoneDetection:
    def test_newcastle(self):
        assumptions = []
        Z, site = _detect_zone("a home in newcastle nsw", assumptions)
        assert Z == 0.11
        assert site == "De"

    def test_sydney(self):
        assumptions = []
        Z, site = _detect_zone("located in sydney", assumptions)
        assert Z == 0.08

    def test_melbourne(self):
        assumptions = []
        Z, site = _detect_zone("in melbourne victoria", assumptions)
        assert Z == 0.08

    def test_brisbane(self):
        assumptions = []
        Z, site = _detect_zone("brisbane queensland", assumptions)
        assert Z == 0.05

    def test_default_newcastle(self):
        assumptions = []
        Z, site = _detect_zone("somewhere in australia", assumptions)
        assert Z == 0.11
        assert len(assumptions) > 0


class TestFullExtraction:
    def test_building1_description(self):
        """Test extraction on Building 1 reference description"""
        desc = ("A 2-storey reinforced concrete residential building in "
                "Newcastle built approximately in 1975. Floor plan "
                "approximately 12 metres by 8 metres.")
        params = _demo_extract(desc)
        assert params['num_storeys'] == 2
        assert params['era'] == 'pre-1990'
        assert params['fc'] == 20.0
        assert params['fy'] == 250.0
        assert params['mu'] == 2.0
        assert params['Z'] == 0.11

    def test_building3_description(self):
        """Test extraction on Building 3 reference description"""
        desc = ("A modern 2-storey RC frame home built in 2018 in "
                "Newcastle, fully compliant with AS1170.4.")
        params = _demo_extract(desc)
        assert params['era'] == 'post-2010'
        assert params['fc'] == 40.0
        assert params['fy'] == 500.0
        assert params['mu'] == 4.0

    def test_all_required_fields_present(self):
        """All required fields must be present in any extraction"""
        required = ['building_name', 'num_storeys', 'storey_height',
                    'num_bays', 'bay_width', 'floor_width', 'fc', 'fy',
                    'col_b', 'col_h', 'beam_b', 'beam_h', 'col_rho',
                    'beam_rho_t', 'beam_rho_c', 'mu', 'Sp', 'Z',
                    'site_class', 'dead_load', 'live_load', 'era',
                    'confidence', 'assumptions']
        params = _demo_extract("a house in australia")
        for field in required:
            assert field in params, f"Missing field: {field}"


class TestValidation:
    def test_valid_params_no_warnings(self):
        params = _demo_extract(
            "2-storey RC frame in Newcastle built 1975, 12m x 8m")
        warnings = validate(params)
        # Should have no out-of-range warnings (assumptions are not warnings)
        range_warnings = [w for w in warnings if 'OUT OF RANGE' in w]
        assert len(range_warnings) == 0

    def test_out_of_range_fc(self):
        params = _demo_extract("a house")
        params['fc'] = 150.0  # unrealistically high
        warnings = validate(params)
        assert any('fc' in w for w in warnings)

    def test_beam_depth_vs_width(self):
        params = _demo_extract("a house")
        params['beam_h'] = 0.20
        params['beam_b'] = 0.30  # depth < width — geometry error
        warnings = validate(params)
        assert any('beam' in w.lower() for w in warnings)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-v'])
