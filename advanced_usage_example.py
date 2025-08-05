#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Thermal Insulation Analysis - Usage Example
====================================================

This script demonstrates how to use the Advanced Thermal Analyzer to:
1. Train models on Kloriz data
2. Predict surface temperatures for complex equipment
3. Generate batch predictions
4. Create analysis reports

Author: Advanced Thermal Analysis System
Version: 3.0
"""

import pandas as pd
import numpy as np
from advanced_thermal_analyzer import AdvancedThermalAnalyzer
import json

def demonstrate_training():
    """
    Demonstrate the training process
    """
    print("🎓 TRAINING DEMONSTRATION")
    print("=" * 50)
    
    # Initialize the analyzer
    analyzer = AdvancedThermalAnalyzer()
    
    # Parse HTML files from Kloriz
    kloriz_data = analyzer.parse_html_files()
    print(f"\n📊 Parsed Data Summary:")
    print(f"   - Total samples: {len(kloriz_data)}")
    print(f"   - Equipment types: {kloriz_data['equipment_type'].unique().tolist()}")
    print(f"   - Temperature range: {kloriz_data['internal_temperature'].min():.1f}°C - {kloriz_data['internal_temperature'].max():.1f}°C")
    
    # Load insulation data
    insulation_data = analyzer.load_insulation_data()
    
    # Prepare training data
    training_data = analyzer.prepare_training_data()
    
    # Train models
    models = analyzer.train_models()
    
    # Save the trained model
    analyzer.save_model("/workspace/advanced_thermal_model.joblib")
    
    # Generate analysis report
    analyzer.generate_analysis_report("/workspace/advanced_thermal_report.html")
    
    return analyzer

def demonstrate_predictions(analyzer):
    """
    Demonstrate predictions for various complex equipment
    """
    print("\n🔮 PREDICTION DEMONSTRATION")
    print("=" * 50)
    
    # Complex equipment examples not typically found in Kloriz
    complex_equipment = [
        {
            'equipment_type': 'Heat Exchanger',
            'internal_temperature': 450.0,
            'ambient_temperature': 25.0,
            'wind_speed': 2.5,
            'total_thickness': 120.0,
            'surface_area': 28.5
        },
        {
            'equipment_type': 'Compressor',
            'internal_temperature': 380.0,
            'ambient_temperature': 30.0,
            'wind_speed': 1.8,
            'total_thickness': 95.0,
            'surface_area': 15.2
        },
        {
            'equipment_type': 'Reactor',
            'internal_temperature': 650.0,
            'ambient_temperature': 22.0,
            'wind_speed': 3.2,
            'total_thickness': 150.0,
            'surface_area': 45.8
        },
        {
            'equipment_type': 'Tank',
            'internal_temperature': 200.0,
            'ambient_temperature': 28.0,
            'wind_speed': 1.2,
            'total_thickness': 80.0,
            'surface_area': 62.3
        }
    ]
    
    predictions = []
    
    for equipment in complex_equipment:
        print(f"\n🔧 Analyzing {equipment['equipment_type']}...")
        print(f"   Internal Temperature: {equipment['internal_temperature']}°C")
        print(f"   Ambient Temperature: {equipment['ambient_temperature']}°C")
        print(f"   Wind Speed: {equipment['wind_speed']} m/s")
        print(f"   Insulation Thickness: {equipment['total_thickness']} mm")
        print(f"   Surface Area: {equipment['surface_area']} m²")
        
        # Make prediction
        result = analyzer.predict_temperature(equipment)
        predictions.append({**equipment, **result})
        
        print(f"   🎯 Predicted Surface Temperature: {result['predicted_surface_temperature']}°C ± {result['confidence_range']}°C")
        print(f"   📊 Model Used: {result['model_used']}")
    
    return predictions

def create_batch_predictions(analyzer):
    """
    Create batch predictions for multiple equipment scenarios
    """
    print("\n📋 BATCH PREDICTION DEMONSTRATION")
    print("=" * 50)
    
    # Generate multiple scenarios
    scenarios = []
    
    # Different turbine types
    turbine_types = ['Gas Turbine GT13E2', 'Steam Turbine ST40', 'Wind Turbine V164']
    for i, turbine in enumerate(turbine_types):
        scenarios.append({
            'scenario_id': f'TURB_{i+1:03d}',
            'equipment_type': 'Turbine V94.2',  # Use known type for prediction
            'description': turbine,
            'internal_temperature': np.random.uniform(400, 600),
            'ambient_temperature': np.random.uniform(15, 35),
            'wind_speed': np.random.uniform(0.5, 5.0),
            'total_thickness': np.random.uniform(80, 160),
            'surface_area': np.random.uniform(20, 50)
        })
    
    # Different compressor types
    compressor_types = ['Centrifugal Compressor', 'Axial Compressor', 'Reciprocating Compressor']
    for i, compressor in enumerate(compressor_types):
        scenarios.append({
            'scenario_id': f'COMP_{i+1:03d}',
            'equipment_type': 'Compressor',
            'description': compressor,
            'internal_temperature': np.random.uniform(300, 500),
            'ambient_temperature': np.random.uniform(20, 40),
            'wind_speed': np.random.uniform(1.0, 4.0),
            'total_thickness': np.random.uniform(70, 140),
            'surface_area': np.random.uniform(10, 30)
        })
    
    # Different valve types
    valve_types = ['Gate Valve', 'Globe Valve', 'Ball Valve', 'Butterfly Valve']
    for i, valve in enumerate(valve_types):
        scenarios.append({
            'scenario_id': f'VALV_{i+1:03d}',
            'equipment_type': 'Valve',
            'description': valve,
            'internal_temperature': np.random.uniform(150, 400),
            'ambient_temperature': np.random.uniform(18, 32),
            'wind_speed': np.random.uniform(0.8, 3.5),
            'total_thickness': np.random.uniform(50, 100),
            'surface_area': np.random.uniform(2, 8)
        })
    
    # Make batch predictions
    batch_results = []
    
    for scenario in scenarios:
        try:
            prediction = analyzer.predict_temperature(scenario)
            result = {
                'Scenario_ID': scenario['scenario_id'],
                'Equipment_Type': scenario['equipment_type'],
                'Description': scenario['description'],
                'Internal_Temp_C': round(scenario['internal_temperature'], 1),
                'Ambient_Temp_C': round(scenario['ambient_temperature'], 1),
                'Wind_Speed_ms': round(scenario['wind_speed'], 1),
                'Thickness_mm': round(scenario['total_thickness'], 1),
                'Surface_Area_m2': round(scenario['surface_area'], 1),
                'Predicted_Surface_Temp_C': prediction['predicted_surface_temperature'],
                'Confidence_Range_C': prediction['confidence_range'],
                'Model_Used': prediction['model_used']
            }
            batch_results.append(result)
            
        except Exception as e:
            print(f"❌ Error predicting for {scenario['scenario_id']}: {str(e)}")
    
    # Save results to Excel
    batch_df = pd.DataFrame(batch_results)
    batch_df.to_excel('/workspace/batch_predictions.xlsx', index=False)
    
    print(f"✅ Batch predictions completed: {len(batch_results)} scenarios")
    print(f"💾 Results saved to: /workspace/batch_predictions.xlsx")
    
    # Display summary statistics
    print(f"\n📊 Prediction Summary:")
    print(f"   Average Predicted Temperature: {batch_df['Predicted_Surface_Temp_C'].mean():.1f}°C")
    print(f"   Temperature Range: {batch_df['Predicted_Surface_Temp_C'].min():.1f}°C - {batch_df['Predicted_Surface_Temp_C'].max():.1f}°C")
    print(f"   Average Confidence Range: ±{batch_df['Confidence_Range_C'].mean():.1f}°C")
    
    return batch_results

def demonstrate_thermal_safety_analysis(analyzer):
    """
    Demonstrate thermal safety analysis for industrial equipment
    """
    print("\n🔥 THERMAL SAFETY ANALYSIS")
    print("=" * 50)
    
    # Safety temperature thresholds
    safety_thresholds = {
        'Safe': 60,      # Below 60°C - Safe for personnel contact
        'Caution': 80,   # 60-80°C - Caution required
        'Warning': 100,  # 80-100°C - Warning level
        'Danger': 120,   # Above 120°C - Dangerous
    }
    
    # Industrial equipment scenarios
    industrial_equipment = [
        {
            'name': 'Main Steam Line',
            'equipment_type': 'Horizontal Pipe',
            'internal_temperature': 540.0,
            'ambient_temperature': 25.0,
            'wind_speed': 2.0,
            'total_thickness': 100.0,
            'surface_area': 25.0
        },
        {
            'name': 'Pressure Vessel',
            'equipment_type': 'Tank',
            'internal_temperature': 320.0,
            'ambient_temperature': 28.0,
            'wind_speed': 1.5,
            'total_thickness': 120.0,
            'surface_area': 40.0
        },
        {
            'name': 'Heat Recovery Unit',
            'equipment_type': 'Heat Exchanger',
            'internal_temperature': 450.0,
            'ambient_temperature': 30.0,
            'wind_speed': 3.0,
            'total_thickness': 90.0,
            'surface_area': 35.0
        }
    ]
    
    safety_analysis = []
    
    for equipment in industrial_equipment:
        prediction = analyzer.predict_temperature(equipment)
        surface_temp = prediction['predicted_surface_temperature']
        
        # Determine safety level
        if surface_temp <= safety_thresholds['Safe']:
            safety_level = 'Safe ✅'
            safety_color = 'green'
        elif surface_temp <= safety_thresholds['Caution']:
            safety_level = 'Caution ⚠️'
            safety_color = 'yellow'
        elif surface_temp <= safety_thresholds['Warning']:
            safety_level = 'Warning ⚠️'
            safety_color = 'orange'
        else:
            safety_level = 'Danger ❌'
            safety_color = 'red'
        
        analysis = {
            'equipment_name': equipment['name'],
            'surface_temperature': surface_temp,
            'safety_level': safety_level,
            'safety_color': safety_color,
            'recommendation': get_safety_recommendation(surface_temp, safety_thresholds)
        }
        
        safety_analysis.append(analysis)
        
        print(f"\n🏭 {equipment['name']}:")
        print(f"   Surface Temperature: {surface_temp}°C")
        print(f"   Safety Level: {safety_level}")
        print(f"   Recommendation: {analysis['recommendation']}")
    
    return safety_analysis

def get_safety_recommendation(temp, thresholds):
    """
    Get safety recommendation based on temperature
    """
    if temp <= thresholds['Safe']:
        return "Equipment is safe for normal operation and personnel contact."
    elif temp <= thresholds['Caution']:
        return "Use caution around equipment. Consider warning signs."
    elif temp <= thresholds['Warning']:
        return "Install warning signs and barriers. Limit personnel access."
    else:
        return "CRITICAL: Install protective barriers and warning systems. Restrict access."

def main():
    """
    Main demonstration function
    """
    print("🚀 ADVANCED THERMAL INSULATION ANALYSIS SYSTEM")
    print("=" * 60)
    print("این سیستم برای پیش‌بینی دمای سطح تجهیزات پیچیده طراحی شده است")
    print("=" * 60)
    
    try:
        # Step 1: Train the model
        analyzer = demonstrate_training()
        
        # Step 2: Make individual predictions
        predictions = demonstrate_predictions(analyzer)
        
        # Step 3: Create batch predictions
        batch_results = create_batch_predictions(analyzer)
        
        # Step 4: Perform safety analysis
        safety_analysis = demonstrate_thermal_safety_analysis(analyzer)
        
        print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Generated Files:")
        print("   - advanced_thermal_model.joblib (Trained model)")
        print("   - advanced_thermal_report.html (Analysis report)")
        print("   - batch_predictions.xlsx (Batch predictions)")
        
        print("\n💡 Usage Tips:")
        print("   1. Load the saved model for future predictions")
        print("   2. Adjust insulation thickness to optimize surface temperature")
        print("   3. Use batch prediction for multiple equipment analysis")
        print("   4. Consider safety recommendations for personnel protection")
        
        return analyzer, predictions, batch_results, safety_analysis
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        return None, None, None, None

if __name__ == "__main__":
    analyzer, predictions, batch_results, safety_analysis = main()