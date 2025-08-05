#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Thermal Predictor - Easy Interface
=========================================

Simple interface for predicting surface temperatures of thermal equipment
using the trained machine learning model.

Usage:
    python3 simple_predictor.py

Author: Advanced Thermal Analysis System
Version: 3.0
"""

from advanced_thermal_analyzer import AdvancedThermalAnalyzer
import pandas as pd

def predict_single_equipment():
    """
    Interactive prediction for a single piece of equipment
    """
    print("🔮 THERMAL SURFACE TEMPERATURE PREDICTOR")
    print("=" * 50)
    print("Please enter your equipment specifications:")
    print()
    
    # Load the trained model
    analyzer = AdvancedThermalAnalyzer()
    if not analyzer.load_model("/workspace/advanced_thermal_model.joblib"):
        print("❌ Model not found. Please run the training first.")
        return
    
    # Load insulation data
    analyzer.load_insulation_data()
    
    # Get user input
    try:
        print("Available equipment types:")
        equipment_types = [
            'Horizontal Pipe', 'Vertical Pipe', 'Horizontal Flat Surface',
            'Vertical Flat Surface', 'Sphere', 'Cube', 'Turbine V94.2',
            'Valve', 'Compressor', 'Heat Exchanger', 'Reactor', 'Tank'
        ]
        
        for i, eq_type in enumerate(equipment_types, 1):
            print(f"  {i}. {eq_type}")
        
        choice = int(input("\nSelect equipment type (number): ")) - 1
        equipment_type = equipment_types[choice] if 0 <= choice < len(equipment_types) else 'Compressor'
        
        internal_temp = float(input("Internal equipment temperature (°C): "))
        ambient_temp = float(input("Ambient temperature (°C): "))
        wind_speed = float(input("Wind speed (m/s): "))
        thickness = float(input("Total insulation thickness (mm): "))
        surface_area = float(input("Total surface area (m²): "))
        
        equipment_data = {
            'equipment_type': equipment_type,
            'internal_temperature': internal_temp,
            'ambient_temperature': ambient_temp,
            'wind_speed': wind_speed,
            'total_thickness': thickness,
            'surface_area': surface_area
        }
        
        print("\n🔄 Calculating...")
        
        # Make prediction
        result = analyzer.predict_temperature(equipment_data)
        
        # Display results
        print("\n" + "=" * 50)
        print("🎯 Prediction Results:")
        print("=" * 50)
        print(f"Equipment Type: {equipment_type}")
        print(f"Predicted Surface Temperature: {result['predicted_surface_temperature']}°C")
        print(f"Confidence Range: ±{result['confidence_range']}°C")
        print(f"Model Used: {result['model_used']}")
        
        # Safety analysis
        surface_temp = result['predicted_surface_temperature']
        if surface_temp <= 60:
            safety = "Safe ✅"
            recommendation = "Equipment is safe for normal operation and personnel contact."
        elif surface_temp <= 80:
            safety = "Caution ⚠️"
            recommendation = "Use caution around equipment. Consider installing warning signs."
        elif surface_temp <= 100:
            safety = "Warning ⚠️"
            recommendation = "Install warning signs and barriers. Limit personnel access."
        else:
            safety = "Danger ❌"
            recommendation = "CRITICAL: Install protective barriers and warning systems. Restrict access."
        
        print(f"\n🛡️ Safety Analysis:")
        print(f"Safety Level: {safety}")
        print(f"Recommendation: {recommendation}")
        
        # Save result
        result_df = pd.DataFrame([{
            'Equipment_Type': equipment_type,
            'Internal_Temp_C': internal_temp,
            'Ambient_Temp_C': ambient_temp,
            'Wind_Speed_ms': wind_speed,
            'Thickness_mm': thickness,
            'Surface_Area_m2': surface_area,
            'Predicted_Surface_Temp_C': result['predicted_surface_temperature'],
            'Confidence_Range_C': result['confidence_range'],
            'Safety_Level': safety,
            'Recommendation': recommendation
        }])
        
        result_df.to_excel('/workspace/prediction_result.xlsx', index=False)
        print(f"\n💾 Results saved to file: /workspace/prediction_result.xlsx")
        
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")

def quick_prediction_examples():
    """
    Quick prediction examples for common equipment
    """
    print("\n🚀 QUICK PREDICTION EXAMPLES")
    print("=" * 50)
    
    analyzer = AdvancedThermalAnalyzer()
    if not analyzer.load_model("/workspace/advanced_thermal_model.joblib"):
        print("❌ Model not found. Please run the training first.")
        return
    
    analyzer.load_insulation_data()
    
    examples = [
        {
            'name': 'Industrial Gas Turbine',
            'equipment_type': 'Turbine V94.2',
            'internal_temperature': 550.0,
            'ambient_temperature': 25.0,
            'wind_speed': 2.0,
            'total_thickness': 120.0,
            'surface_area': 40.0
        },
        {
            'name': 'Centrifugal Compressor',
            'equipment_type': 'Compressor',
            'internal_temperature': 350.0,
            'ambient_temperature': 30.0,
            'wind_speed': 1.5,
            'total_thickness': 100.0,
            'surface_area': 18.0
        },
        {
            'name': 'Heat Exchanger',
            'equipment_type': 'Heat Exchanger',
            'internal_temperature': 400.0,
            'ambient_temperature': 28.0,
            'wind_speed': 2.5,
            'total_thickness': 90.0,
            'surface_area': 25.0
        }
    ]
    
    for example in examples:
        print(f"\n🔧 {example['name']}:")
        result = analyzer.predict_temperature(example)
        print(f"   Surface Temperature: {result['predicted_surface_temperature']}°C ± {result['confidence_range']}°C")

def main():
    """
    Main function
    """
    print("🌡️ Thermal Insulation Surface Temperature Prediction System")
    print("=" * 60)
    print("This system uses Kloriz software data and machine learning")
    print("to predict surface temperatures of complex equipment")
    print("=" * 60)
    
    while True:
        print("\nChoose an option:")
        print("1. Predict for single equipment")
        print("2. Quick examples")
        print("3. Exit")
        
        try:
            choice = input("\nYour choice (1-3): ").strip()
            
            if choice == '1':
                predict_single_equipment()
            elif choice == '2':
                quick_prediction_examples()
            elif choice == '3':
                print("Goodbye! 👋")
                break
            else:
                print("Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()