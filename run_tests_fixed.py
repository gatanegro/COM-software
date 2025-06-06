"""
COM Framework Test Runner with Fixed Modules

This script runs the test suite for the COM Framework using the fixed modules.
"""

import unittest
import sys
import os
import time
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the fixed modules
from com_framework_core_fixed import LZModule, OctaveModule
from com_visualization_fixed import VisualizationModule
from com_analysis_fixed import (
    MathematicalAnalysisModule,
    PatternRecognitionModule,
    StatisticalAnalysisModule
)

class TestLZModule(unittest.TestCase):
    """Test cases for the LZ Module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lz_module = LZModule()
        
    def test_lz_constant(self):
        """Test that the LZ constant is correctly defined."""
        self.assertAlmostEqual(self.lz_module.LZ, 1.23498, places=5)
        
    def test_recursive_wave_function(self):
        """Test the recursive wave function."""
        # Test with known values
        self.assertAlmostEqual(
            self.lz_module.recursive_wave_function(0),
            0 + 1,  # sin(0) + e^0
            places=5
        )
        self.assertAlmostEqual(
            self.lz_module.recursive_wave_function(1),
            np.sin(1) + np.exp(-1),
            places=5
        )
        
    def test_derive_lz(self):
        """Test LZ derivation through iteration."""
        # Derive LZ from different starting points
        for start in [0.5, 1.0, 1.5, 2.0]:
            derived_lz, sequence, iterations = self.lz_module.derive_lz(
                initial_value=start,
                max_iterations=100,
                precision=1e-6
            )
            
            # Check that it converges to LZ
            self.assertAlmostEqual(derived_lz, self.lz_module.LZ, places=4)
            
            # Check that sequence has correct length
            self.assertEqual(len(sequence), iterations + 1)
            
            # Check that sequence starts with initial value
            self.assertEqual(sequence[0], start)
            
            # Check that sequence ends with derived LZ
            self.assertAlmostEqual(sequence[-1], derived_lz, places=4)
            
    def test_verify_lz(self):
        """Test verification that LZ is a fixed point."""
        self.assertTrue(self.lz_module.verify_lz())
        
    def test_stability_at_point(self):
        """Test stability calculation at various points."""
        # LZ should be stable (derivative magnitude < 1)
        self.assertLess(self.lz_module.stability_at_point(self.lz_module.LZ), 1)
        
        # Test a known unstable point
        self.assertGreater(self.lz_module.stability_at_point(0), 1)
        
    def test_is_stable_fixed_point(self):
        """Test stable fixed point detection."""
        # LZ should be a stable fixed point
        self.assertTrue(self.lz_module.is_stable_fixed_point(self.lz_module.LZ))
        
        # 0 is a fixed point of sin(x) + e^(-x) but not stable
        self.assertFalse(self.lz_module.is_stable_fixed_point(0))
        
    def test_lz_scaling(self):
        """Test LZ-based scaling."""
        # Test scaling with different octaves
        base_value = 1.0
        for octave in range(-3, 4):
            scaled = self.lz_module.lz_scaling(base_value, octave)
            expected = base_value * (self.lz_module.LZ ** octave)
            self.assertAlmostEqual(scaled, expected, places=5)
            
    def test_find_fixed_points(self):
        """Test finding fixed points in a range."""
        # Find fixed points in [0, 5]
        fixed_points = self.lz_module.find_fixed_points(0, 5, 0.01)
        
        # Should find at least LZ
        self.assertGreaterEqual(len(fixed_points), 1)
        
        # LZ should be in the list
        self.assertTrue(any(abs(fp - self.lz_module.LZ) < 0.01 for fp in fixed_points))
        
        # All points should be fixed points
        for fp in fixed_points:
            self.assertAlmostEqual(
                self.lz_module.recursive_wave_function(fp),
                fp,
                places=2
            )
            
    def test_plot_recursive_function(self):
        """Test plotting the recursive function."""
        fig = self.lz_module.plot_recursive_function()
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_plot_convergence(self):
        """Test plotting convergence to LZ."""
        fig = self.lz_module.plot_convergence()
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_stability_analysis(self):
        """Test stability analysis plotting."""
        fig = self.lz_module.stability_analysis()
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_hqs_threshold_function(self):
        """Test HQS threshold function."""
        # Below threshold should return 0
        self.assertEqual(
            self.lz_module.hqs_threshold_function(0),
            0
        )
        
        # Above threshold should return 1
        self.assertEqual(
            self.lz_module.hqs_threshold_function(2 * np.pi * self.lz_module.HQS + 0.1),
            1
        )
        
    def test_recursive_hqs(self):
        """Test recursive HQS function."""
        # Test with various phase differences
        phase_diffs = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi]
        results = self.lz_module.recursive_hqs(phase_diffs, depth=3)
        
        # Should return a list of the same length
        self.assertEqual(len(results), len(phase_diffs))
        
        # All values should be between 0 and 1
        for result in results:
            self.assertGreaterEqual(result, 0)
            self.assertLessEqual(result, 1)


class TestOctaveModule(unittest.TestCase):
    """Test cases for the Octave Module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lz_module = LZModule()
        self.octave_module = OctaveModule(self.lz_module)
        
    def test_octave_reduction(self):
        """Test octave reduction function."""
        # Test known values
        test_cases = [
            (0, 9),    # 0 -> 9
            (1, 1),    # 1 -> 1
            (9, 9),    # 9 -> 9
            (10, 1),   # 10 -> 1
            (27, 9),   # 27 -> 9
            (123, 6)   # 123 -> 6
        ]
        
        for input_val, expected in test_cases:
            self.assertEqual(
                self.octave_module.octave_reduction(input_val),
                expected
            )
            
    def test_octave_reduction_sequence(self):
        """Test octave reduction on a sequence."""
        sequence = [1, 2, 3, 10, 11, 12, 19, 20, 21]
        expected = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        
        result = self.octave_module.octave_reduction_sequence(sequence)
        self.assertEqual(result, expected)
        
    def test_lz_based_octave(self):
        """Test LZ-based octave mapping."""
        # Test with powers of LZ
        for i in range(-3, 4):
            value = self.lz_module.LZ ** i
            octave = self.octave_module.lz_based_octave(value)
            
            # Should be between 0 and 1
            self.assertGreaterEqual(octave, 0)
            self.assertLess(octave, 1)
            
    def test_collatz_octave_transform(self):
        """Test Collatz-Octave transformation."""
        # Test with known starting values
        n = 27
        key = 1
        steps = 20
        
        result = self.octave_module.collatz_octave_transform(n, key, steps)
        
        # Should return a list
        self.assertIsInstance(result, list)
        
        # All values should be octaves (1-9)
        for val in result:
            self.assertGreaterEqual(val, 1)
            self.assertLessEqual(val, 9)
            
        # Different keys should produce different results
        result2 = self.octave_module.collatz_octave_transform(n, key + 1, steps)
        self.assertNotEqual(result, result2)
        
    def test_octave_distribution(self):
        """Test octave distribution calculation."""
        sequence = list(range(1, 100))
        distribution = self.octave_module.octave_distribution(sequence)
        
        # Should return a dictionary with keys 1-9
        self.assertIsInstance(distribution, dict)
        for i in range(1, 10):
            self.assertIn(i, distribution)
            
        # Sum of counts should equal length of sequence
        self.assertEqual(sum(distribution.values()), len(sequence))
        
    def test_plot_octave_distribution(self):
        """Test plotting octave distribution."""
        sequence = list(range(1, 100))
        fig = self.octave_module.plot_octave_distribution(sequence)
        self.assertIsNotNone(fig)
        plt.close(fig)


class TestVisualizationModule(unittest.TestCase):
    """Test cases for the Visualization Module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lz_module = LZModule()
        self.octave_module = OctaveModule(self.lz_module)
        self.viz_module = VisualizationModule(self.lz_module, self.octave_module)
        
    def test_create_dashboard(self):
        """Test dashboard creation."""
        fig = self.viz_module.create_dashboard()
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_create_energy_pattern_visualization(self):
        """Test energy pattern visualization."""
        # Use smaller size for faster testing
        fig = self.viz_module.create_energy_pattern_visualization(size=50, iterations=5)
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_create_cryptographic_visualization(self):
        """Test cryptographic visualization."""
        fig = self.viz_module.create_cryptographic_visualization(text="TEST")
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_create_interactive_lz_explorer(self):
        """Test interactive LZ explorer."""
        fig = self.viz_module.create_interactive_lz_explorer()
        self.assertIsNotNone(fig)
        plt.close(fig)
        
    def test_create_lz_convergence_animation(self):
        """Test LZ convergence animation."""
        anim = self.viz_module.create_lz_convergence_animation(
            initial_values=[0.5, 1.0, 1.5],
            max_iterations=10
        )
        self.assertIsNotNone(anim)
        plt.close(anim._fig)
        
    def test_save_all_visualizations(self):
        """Test saving all visualizations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_paths = self.viz_module.save_all_visualizations(temp_dir)
            
            # Should return a dictionary with file paths
            self.assertIsInstance(file_paths, dict)
            
            # All files should exist
            for path in file_paths.values():
                self.assertTrue(os.path.exists(path))


class TestMathematicalAnalysisModule(unittest.TestCase):
    """Test cases for the Mathematical Analysis Module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lz_module = LZModule()
        self.octave_module = OctaveModule(self.lz_module)
        self.math_module = MathematicalAnalysisModule(self.lz_module, self.octave_module)
        
    def test_analyze_fixed_points(self):
        """Test fixed points analysis."""
        # Use smaller range for faster testing
        results = self.math_module.analyze_fixed_points(0, 3, 0.1)
        
        # Should return a dictionary with expected keys
        self.assertIsInstance(results, dict)
        expected_keys = ['fixed_points', 'count', 'stability', 'basin_sizes', 'convergence_rates']
        for key in expected_keys:
            self.assertIn(key, results)
            
        # Should find at least one fixed point (LZ)
        self.assertGreaterEqual(results['count'], 1)
        
    def test_analyze_lz_powers(self):
        """Test LZ powers analysis."""
        results = self.math_module.analyze_lz_powers(max_power=5)
        
        # Should return a dictionary with expected keys
        self.assertIsInstance(results, dict)
        expected_keys = ['powers', 'octaves', 'pattern_length', 'ratios']
        for key in expected_keys:
            self.assertIn(key, results)
            
        # Powers list should have correct length
        self.assertEqual(len(results['powers']), 11)  # -5 to 5
        
    def test_analyze_octave_distribution(self):
        """Test octave distribution analysis."""
        # Use smaller range for faster testing
        results = self.math_module.analyze_octave_distribution(1, 100)
        
        # Should return a dictionary with expected keys
        self.assertIsInstance(results, dict)
        expected_keys = ['counts', 'chi2_stat', 'p_value', 'is_uniform', 'autocorrelation']
        for key in expected_keys:
            self.assertIn(key, results)
            
        # Counts should include all octaves
        for i in range(1, 10):
            self.assertIn(i, results['counts'])
            
    def test_analyze_collatz_octave_properties(self):
        """Test Collatz-Octave properties analysis."""
        # Use smaller range for faster testing
        results = self.math_module.analyze_collatz_octave_properties(max_n=20)
        
        # Should return a dictionary with expected keys
        self.assertIsInstance(results, dict)
        expected_keys = ['cycle_lengths', 'sequence_lengths', 'unique_octaves', 'entropy']
        for key in expected_keys:
            self.assertIn(key, results)
            
        # Lists should have correct length
        self.assertEqual(len(results['cycle_lengths']), 20)
        
    def test_analyze_lz_hqs_relationship(self):
        """Test LZ-HQS relationship analysis."""
        # Use fewer points for faster testing
        results = self.math_module.analyze_lz_hqs_relationship(points=100)
        
        # Should return a dictionary with expected keys
        self.assertIsInstance(results, dict)
        expected_keys = ['x_values', 'f_values', 'derivatives', 'stability_crossings', 'hqs_values']
        for key in expected_keys:
            self.assertIn(key, results)
            
        # Lists should have correct length
        self.assertEqual(len(results['x_values']), 100)
        
    def test_analyze_octave_scaling(self):
        """Test octave scaling analysis."""
        results = self.math_module.analyze_octave_scaling()
        
        # Should return a dictionary with expected keys
        self.assertIsInstance(results, dict)
        expected_keys = ['scaled_values', 'ratios', 'octave_values', 'pattern_length']
        for key in expected_keys:
            self.assertIn(key, results)
            
        # Lists should have correct length
        self.assertEqual(len(results['scaled_values']), 8)
        self.assertEqual(len(results['ratios']), 7)


class TestPatternRecognitionModule(unittest.TestCase):
    """Test cases for the Pattern Recognition Module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lz_module = LZModule()
        self.octave_module = OctaveModule(self.lz_module)
        self.pattern_module = PatternRecognitionModule(self.lz_module, self.octave_module)
        
    def test_detect_octave_patterns(self):
        """Test octave pattern detection."""
        # Create a sequence with known patterns
        data = [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6]
        results = self.pattern_module.detect_octave_patterns(data)
        
        # Should return a dictionary with expected keys
        self.assertIsInstance(results, dict)
        expected_keys = ['octaves', 'patterns', 'frequency_distribution', 'transition_matrix', 'rhythm']
        for key in expected_keys:
            self.assertIn(key, results)
            
        # Should detect at least one pattern
        self.assertGreaterEqual(len(re
(Content truncated due to size limit. Use line ranges to read in chunks)