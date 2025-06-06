"""
COM Framework Test Runner

This script runs the test suite for the COM Framework and generates a test report.
"""

import unittest
import sys
import os
import time
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the test module
from com_tests import (
    TestLZModule,
    TestOctaveModule,
    TestVisualizationModule,
    TestMathematicalAnalysisModule,
    TestPatternRecognitionModule,
    TestStatisticalAnalysisModule
)

def run_tests_with_report():
    """Run all tests and generate a detailed report."""
    # Create output directory for test results
    os.makedirs('test_results', exist_ok=True)
    
    # Open report file
    report_path = os.path.join('test_results', f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(report_path, 'w') as report_file:
        # Write report header
        report_file.write("COM Framework Test Report\n")
        report_file.write("=======================\n\n")
        report_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Create test suite
        test_suite = unittest.TestSuite()
        
        # Add test cases
        test_classes = [
            TestLZModule,
            TestOctaveModule,
            TestVisualizationModule,
            TestMathematicalAnalysisModule,
            TestPatternRecognitionModule,
            TestStatisticalAnalysisModule
        ]
        
        # Track overall statistics
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_errors = 0
        
        # Run tests for each module
        for test_class in test_classes:
            report_file.write(f"\nTesting {test_class.__name__}\n")
            report_file.write("-" * (len(f"Testing {test_class.__name__}") + 1) + "\n")
            
            # Create test suite for this class
            class_suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            total_tests += class_suite.countTestCases()
            
            # Create a result collector
            result = unittest.TestResult()
            
            # Run tests and time execution
            start_time = time.time()
            class_suite.run(result)
            end_time = time.time()
            
            # Update statistics
            passed = result.testsRun - len(result.failures) - len(result.errors)
            total_passed += passed
            total_failed += len(result.failures)
            total_errors += len(result.errors)
            
            # Write results to report
            report_file.write(f"Tests run: {result.testsRun}\n")
            report_file.write(f"Passed: {passed}\n")
            report_file.write(f"Failed: {len(result.failures)}\n")
            report_file.write(f"Errors: {len(result.errors)}\n")
            report_file.write(f"Time: {end_time - start_time:.2f} seconds\n\n")
            
            # Write details of failures
            if result.failures:
                report_file.write("Failures:\n")
                for i, (test, traceback) in enumerate(result.failures):
                    report_file.write(f"  {i+1}. {test}\n")
                    report_file.write(f"     {traceback.split('Traceback')[0].strip()}\n\n")
            
            # Write details of errors
            if result.errors:
                report_file.write("Errors:\n")
                for i, (test, traceback) in enumerate(result.errors):
                    report_file.write(f"  {i+1}. {test}\n")
                    report_file.write(f"     {traceback.split('Traceback')[0].strip()}\n\n")
        
        # Write summary
        report_file.write("\nSummary\n")
        report_file.write("=======\n")
        report_file.write(f"Total tests: {total_tests}\n")
        report_file.write(f"Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)\n")
        report_file.write(f"Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)\n")
        report_file.write(f"Errors: {total_errors} ({total_errors/total_tests*100:.1f}%)\n")
        
        # Overall result
        if total_failed == 0 and total_errors == 0:
            report_file.write("\nOVERALL RESULT: PASS\n")
        else:
            report_file.write("\nOVERALL RESULT: FAIL\n")
    
    return report_path

if __name__ == "__main__":
    report_path = run_tests_with_report()
    print(f"Test report generated: {report_path}")
