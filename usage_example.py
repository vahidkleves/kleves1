#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage Example for Thermal Insulation Analysis System
Demonstrates how to use the ThermalInsulationAnalyzer class
"""

from thermal_insulation_analyzer import ThermalInsulationAnalyzer
import pandas as pd

def example_basic_usage():
    """Basic usage example"""
    
    print("🔥 Basic Usage Example")
    print("=" * 40)
    
    # Initialize the analyzer
    analyzer = ThermalInsulationAnalyzer()
    
    # Load data from files
    print("\n1️⃣ Loading data...")
    analyzer.parse_kloriz_html_files("a*.html")  # Load HTML files
    analyzer.load_insulation_data("inputdata.xlsx")  # Load Excel data
    
    # Train the machine learning model
    print("\n2️⃣ Training model...")
    analyzer.train_model()
    
    # Make a prediction for new equipment
    print("\n3️⃣ Making prediction...")
    
    new_equipment = {
        'internal_temperature': 500.0,      # °C
        'ambient_temperature': 30.0,        # °C
        'wind_speed': 2.5,                  # m/s
        'total_insulation_thickness': 100.0, # mm
        'total_surface_area': 20.0,         # m²
        'average_density': 125.0,           # kg/m³
        'average_thermal_conductivity': 0.043, # W/m·K
        'average_convection_coefficient': 18.0, # W/m²·K
        'number_of_layers': 3,
        'equipment_type': 'Complex Turbine',
        'dominant_insulation_type': 'Rock Wool'
    }
    
    result = analyzer.predict_surface_temperature(new_equipment)
    
    print(f"\n✅ Prediction Results:")
    print(f"   Surface Temperature: {result['predicted_surface_temperature']} °C")
    print(f"   Confidence Interval: {result['confidence_interval'][0]} - {result['confidence_interval'][1]} °C")
    
    # Save the trained model
    print("\n4️⃣ Saving model...")
    analyzer.save_model("my_thermal_model.joblib")
    
    # Generate report
    print("\n5️⃣ Generating report...")
    analyzer.generate_report("my_thermal_report.html")
    
    return analyzer

def example_batch_predictions():
    """Example of batch predictions for multiple equipment"""
    
    print("\n🔄 Batch Predictions Example")
    print("=" * 40)
    
    # Load existing model
    analyzer = ThermalInsulationAnalyzer()
    
    # If model exists, load it; otherwise train new one
    try:
        analyzer.load_model("my_thermal_model.joblib")
        print("✅ Loaded existing model")
    except:
        print("🔄 Training new model...")
        analyzer.parse_kloriz_html_files("a*.html")
        analyzer.load_insulation_data("inputdata.xlsx")
        analyzer.train_model()
    
    # Define multiple equipment for batch prediction
    equipment_list = [
        {
            'name': 'Compressor Unit A',
            'internal_temperature': 350.0,
            'ambient_temperature': 25.0,
            'wind_speed': 1.5,
            'total_insulation_thickness': 80.0,
            'total_surface_area': 12.0,
            'average_density': 130.0,
            'average_thermal_conductivity': 0.041,
            'average_convection_coefficient': 15.0,
            'number_of_layers': 2,
            'equipment_type': 'Compressor',
            'dominant_insulation_type': 'Cerablanket'
        },
        {
            'name': 'Turbine Casing B',
            'internal_temperature': 600.0,
            'ambient_temperature': 35.0,
            'wind_speed': 4.0,
            'total_insulation_thickness': 120.0,
            'total_surface_area': 45.0,
            'average_density': 110.0,
            'average_thermal_conductivity': 0.039,
            'average_convection_coefficient': 20.0,
            'number_of_layers': 3,
            'equipment_type': 'Complex Turbine',
            'dominant_insulation_type': 'Silika Needeled Mat'
        },
        {
            'name': 'Valve Assembly C',
            'internal_temperature': 280.0,
            'ambient_temperature': 20.0,
            'wind_speed': 0.8,
            'total_insulation_thickness': 60.0,
            'total_surface_area': 2.5,
            'average_density': 140.0,
            'average_thermal_conductivity': 0.044,
            'average_convection_coefficient': 12.0,
            'number_of_layers': 1,
            'equipment_type': 'Valve',
            'dominant_insulation_type': 'Rock Wool'
        }
    ]
    
    # Make predictions for all equipment
    results = []
    
    for equipment in equipment_list:
        name = equipment.pop('name')  # Remove name from prediction data
        
        print(f"\n🔮 Predicting for: {name}")
        result = analyzer.predict_surface_temperature(equipment)
        
        results.append({
            'Equipment_Name': name,
            'Predicted_Surface_Temperature': result['predicted_surface_temperature'],
            'Confidence_Lower': result['confidence_interval'][0],
            'Confidence_Upper': result['confidence_interval'][1],
            'Internal_Temperature': equipment['internal_temperature'],
            'Ambient_Temperature': equipment['ambient_temperature']
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 Batch Prediction Results:")
    print(results_df.to_string(index=False))
    
    # Save results to Excel
    results_df.to_excel('batch_predictions.xlsx', index=False)
    print(f"\n💾 Results saved to: batch_predictions.xlsx")
    
    return results_df

def example_model_comparison():
    """Example comparing different insulation configurations"""
    
    print("\n🔬 Insulation Comparison Example")
    print("=" * 40)
    
    analyzer = ThermalInsulationAnalyzer()
    
    # Load or train model
    try:
        analyzer.load_model("my_thermal_model.joblib")
    except:
        analyzer.parse_kloriz_html_files("a*.html")
        analyzer.load_insulation_data("inputdata.xlsx")
        analyzer.train_model()
    
    # Base equipment parameters
    base_equipment = {
        'internal_temperature': 450.0,
        'ambient_temperature': 25.0,
        'wind_speed': 3.0,
        'total_surface_area': 15.0,
        'average_density': 125.0,
        'average_convection_coefficient': 16.0,
        'number_of_layers': 2,
        'equipment_type': 'Complex Turbine'
    }
    
    # Test different insulation configurations
    configurations = [
        {
            'name': 'Thin Cerablanket',
            'total_insulation_thickness': 50.0,
            'average_thermal_conductivity': 0.045,
            'dominant_insulation_type': 'Cerablanket'
        },
        {
            'name': 'Medium Rock Wool',
            'total_insulation_thickness': 80.0,
            'average_thermal_conductivity': 0.042,
            'dominant_insulation_type': 'Rock Wool'
        },
        {
            'name': 'Thick Silika Mat',
            'total_insulation_thickness': 120.0,
            'average_thermal_conductivity': 0.038,
            'dominant_insulation_type': 'Silika Needeled Mat'
        },
        {
            'name': 'Multi-layer Needeled',
            'total_insulation_thickness': 100.0,
            'average_thermal_conductivity': 0.040,
            'dominant_insulation_type': 'Needeled Mat'
        }
    ]
    
    comparison_results = []
    
    for config in configurations:
        # Combine base equipment with configuration
        equipment = {**base_equipment, **{k: v for k, v in config.items() if k != 'name'}}
        
        result = analyzer.predict_surface_temperature(equipment)
        
        temperature_reduction = equipment['internal_temperature'] - result['predicted_surface_temperature']
        efficiency = (temperature_reduction / (equipment['internal_temperature'] - equipment['ambient_temperature'])) * 100
        
        comparison_results.append({
            'Configuration': config['name'],
            'Thickness_mm': config['total_insulation_thickness'],
            'Thermal_Conductivity': config['average_thermal_conductivity'],
            'Predicted_Surface_Temp': result['predicted_surface_temperature'],
            'Temperature_Reduction': temperature_reduction,
            'Insulation_Efficiency_%': efficiency
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    print(f"\n📊 Insulation Configuration Comparison:")
    print(comparison_df.to_string(index=False, float_format='%.2f'))
    
    # Find best configuration
    best_config = comparison_df.loc[comparison_df['Insulation_Efficiency_%'].idxmax()]
    print(f"\n🏆 Best Configuration: {best_config['Configuration']}")
    print(f"   Efficiency: {best_config['Insulation_Efficiency_%']:.1f}%")
    print(f"   Surface Temperature: {best_config['Predicted_Surface_Temp']:.1f} °C")
    
    # Save comparison results
    comparison_df.to_excel('insulation_comparison.xlsx', index=False)
    print(f"\n💾 Comparison saved to: insulation_comparison.xlsx")
    
    return comparison_df

def main():
    """Main function to run all examples"""
    
    print("🌟 Thermal Insulation Analysis System - Usage Examples")
    print("=" * 60)
    
    try:
        # Run basic usage example
        analyzer = example_basic_usage()
        
        # Run batch predictions example
        batch_results = example_batch_predictions()
        
        # Run model comparison example
        comparison_results = example_model_comparison()
        
        print(f"\n🎉 All examples completed successfully!")
        print(f"\n📄 Generated files:")
        print(f"   - my_thermal_model.joblib (Trained model)")
        print(f"   - my_thermal_report.html (Analysis report)")
        print(f"   - batch_predictions.xlsx (Batch prediction results)")
        print(f"   - insulation_comparison.xlsx (Configuration comparison)")
        
    except Exception as e:
        print(f"❌ Error running examples: {str(e)}")
        print(f"💡 Make sure to run 'python create_sample_data.py' first to create sample data files.")

if __name__ == "__main__":
    main()