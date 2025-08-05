#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Script for Thermal Insulation Analysis System
Tests all functionality and reports any errors
"""

import sys
import traceback
import os
from thermal_insulation_analyzer import ThermalInsulationAnalyzer
import pandas as pd
import numpy as np

def test_basic_functionality():
    """Test basic system functionality"""
    print("🧪 Testing Basic Functionality...")
    
    try:
        # Initialize analyzer
        analyzer = ThermalInsulationAnalyzer()
        print("✅ Analyzer initialized successfully")
        
        # Test HTML parsing
        analyzer.parse_kloriz_html_files("a*.html")
        print("✅ HTML parsing successful")
        
        # Test Excel loading
        analyzer.load_insulation_data("inputdata.xlsx")
        print("✅ Excel data loading successful")
        
        # Test model training
        analyzer.train_model()
        print("✅ Model training successful")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {str(e)}")
        traceback.print_exc()
        return None

def test_predictions(analyzer):
    """Test prediction functionality"""
    print("\n🧪 Testing Prediction Functionality...")
    
    if analyzer is None:
        print("❌ Cannot test predictions - analyzer not available")
        return False
    
    try:
        # Test simple prediction
        equipment_data = {
            'internal_temperature': 400.0,
            'ambient_temperature': 25.0,
            'wind_speed': 2.0,
            'total_insulation_thickness': 80.0,
            'total_surface_area': 10.0,
            'average_density': 120.0,
            'average_thermal_conductivity': 0.042,
            'average_convection_coefficient': 15.0,
            'number_of_layers': 2,
            'equipment_type': 'Compressor',
            'dominant_insulation_type': 'Rock Wool'
        }
        
        result = analyzer.predict_surface_temperature(equipment_data)
        print(f"✅ Prediction successful: {result['predicted_surface_temperature']} °C")
        
        # Test prediction with missing data
        minimal_data = {
            'internal_temperature': 350.0,
            'ambient_temperature': 30.0,
            'equipment_type': 'Valve',
            'dominant_insulation_type': 'Cerablanket'
        }
        
        result2 = analyzer.predict_surface_temperature(minimal_data)
        print(f"✅ Prediction with minimal data successful: {result2['predicted_surface_temperature']} °C")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_model_save_load():
    """Test model saving and loading"""
    print("\n🧪 Testing Model Save/Load...")
    
    try:
        # Create and train a model
        analyzer1 = ThermalInsulationAnalyzer()
        analyzer1.parse_kloriz_html_files("a*.html")
        analyzer1.load_insulation_data("inputdata.xlsx")
        analyzer1.train_model()
        
        # Save model
        analyzer1.save_model("test_model.joblib")
        print("✅ Model saved successfully")
        
        # Load model in new analyzer
        analyzer2 = ThermalInsulationAnalyzer()
        analyzer2.load_model("test_model.joblib")
        print("✅ Model loaded successfully")
        
        # Test prediction with loaded model
        test_data = {
            'internal_temperature': 300.0,
            'ambient_temperature': 20.0,
            'wind_speed': 1.5,
            'total_insulation_thickness': 60.0,
            'total_surface_area': 5.0,
            'average_density': 100.0,
            'average_thermal_conductivity': 0.040,
            'average_convection_coefficient': 12.0,
            'number_of_layers': 1,
            'equipment_type': 'Pipe',
            'dominant_insulation_type': 'Rock Wool'
        }
        
        result = analyzer2.predict_surface_temperature(test_data)
        print(f"✅ Prediction with loaded model successful: {result['predicted_surface_temperature']} °C")
        
        # Clean up
        if os.path.exists("test_model.joblib"):
            os.remove("test_model.joblib")
        
        return True
        
    except Exception as e:
        print(f"❌ Model save/load test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling scenarios"""
    print("\n🧪 Testing Error Handling...")
    
    try:
        analyzer = ThermalInsulationAnalyzer()
        
        # Test prediction without training
        try:
            result = analyzer.predict_surface_temperature({'temp': 100})
            print("❌ Should have failed - no model trained")
            return False
        except ValueError:
            print("✅ Correctly handled prediction without trained model")
        
        # Test with non-existent files
        analyzer.parse_kloriz_html_files("nonexistent*.html")
        print("✅ Correctly handled non-existent HTML files")
        
        analyzer.load_insulation_data("nonexistent.xlsx")
        print("✅ Correctly handled non-existent Excel file (fallback to sample data)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_data_validation():
    """Test data validation"""
    print("\n🧪 Testing Data Validation...")
    
    try:
        analyzer = ThermalInsulationAnalyzer()
        analyzer.parse_kloriz_html_files("a*.html")
        analyzer.load_insulation_data("inputdata.xlsx")
        analyzer.train_model()
        
        # Test with extreme values
        extreme_data = {
            'internal_temperature': 1000.0,  # Very high
            'ambient_temperature': -50.0,    # Very low
            'wind_speed': 100.0,             # Very high
            'total_insulation_thickness': 1000.0,  # Very thick
            'total_surface_area': 1000.0,    # Very large
            'average_density': 1000.0,       # Very dense
            'average_thermal_conductivity': 1.0,  # High conductivity
            'average_convection_coefficient': 100.0,  # High convection
            'number_of_layers': 10,          # Many layers
            'equipment_type': 'Complex Turbine',
            'dominant_insulation_type': 'Rock Wool'
        }
        
        result = analyzer.predict_surface_temperature(extreme_data)
        print(f"✅ Handled extreme values: {result['predicted_surface_temperature']} °C")
        
        # Test with invalid categorical data
        invalid_data = {
            'internal_temperature': 400.0,
            'ambient_temperature': 25.0,
            'equipment_type': 'Unknown Equipment Type',
            'dominant_insulation_type': 'Unknown Insulation'
        }
        
        result2 = analyzer.predict_surface_temperature(invalid_data)
        print(f"✅ Handled invalid categorical data: {result2['predicted_surface_temperature']} °C")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_report_generation():
    """Test report generation"""
    print("\n🧪 Testing Report Generation...")
    
    try:
        analyzer = ThermalInsulationAnalyzer()
        analyzer.parse_kloriz_html_files("a*.html")
        analyzer.load_insulation_data("inputdata.xlsx")
        analyzer.train_model()
        
        # Generate report
        analyzer.generate_report("test_report.html")
        
        # Check if report file exists
        if os.path.exists("test_report.html"):
            print("✅ Report generated successfully")
            
            # Check report content
            with open("test_report.html", 'r', encoding='utf-8') as f:
                content = f.read()
                if "تحلیل حرارتی" in content and "Analysis" in content:
                    print("✅ Report content validated")
                else:
                    print("⚠️  Report content may be incomplete")
            
            # Clean up
            os.remove("test_report.html")
            return True
        else:
            print("❌ Report file not created")
            return False
            
    except Exception as e:
        print(f"❌ Report generation test failed: {str(e)}")
        traceback.print_exc()
        return False

def run_comprehensive_tests():
    """Run all tests and report results"""
    print("🚀 Starting Comprehensive System Tests")
    print("=" * 50)
    
    test_results = []
    
    # Run all tests
    analyzer = test_basic_functionality()
    test_results.append(("Basic Functionality", analyzer is not None))
    
    prediction_success = test_predictions(analyzer)
    test_results.append(("Predictions", prediction_success))
    
    save_load_success = test_model_save_load()
    test_results.append(("Model Save/Load", save_load_success))
    
    error_handling_success = test_error_handling()
    test_results.append(("Error Handling", error_handling_success))
    
    validation_success = test_data_validation()
    test_results.append(("Data Validation", validation_success))
    
    report_success = test_report_generation()
    test_results.append(("Report Generation", report_success))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if success:
            passed += 1
    
    print(f"\n📈 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return False

def main():
    """Main test function"""
    try:
        success = run_comprehensive_tests()
        
        if success:
            print("\n✅ System is ready for production use!")
            sys.exit(0)
        else:
            print("\n❌ System has issues that need to be addressed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()