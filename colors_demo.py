#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colors Thermal Analysis System - Demonstration Script
نمایش سیستم تحلیل حرارتی نرم‌افزار کلورز

This script demonstrates the capabilities of the Colors thermal analysis system
with sample data and automated predictions.
"""

import pandas as pd
import numpy as np
import os
from colors_thermal_analyzer import ColorsThermalMLSystem, ColorsInsulationDatabase
import matplotlib.pyplot as plt
import seaborn as sns

def create_sample_colors_html():
    """Create sample Colors HTML files for demonstration"""
    
    sample_data = [
        {
            'file_name': 'sample_1.html',
            'equipment_type': 'Turbine V94.2',
            'internal_temperature': 558.4,
            'ambient_temperature': 20.5,
            'wind_speed': 6.3,
            'insulation_thickness': 94.6,
            'surface_area': 37.91,
            'insulation_type': 'Cerablanket'
        },
        {
            'file_name': 'sample_2.html',
            'equipment_type': 'Horizontal Pipe',
            'internal_temperature': 450.0,
            'ambient_temperature': 25.0,
            'wind_speed': 4.2,
            'insulation_thickness': 75.0,
            'surface_area': 15.5,
            'insulation_type': 'Mineral Wool'
        },
        {
            'file_name': 'sample_3.html',
            'equipment_type': 'Valve',
            'internal_temperature': 380.0,
            'ambient_temperature': 30.0,
            'wind_speed': 2.8,
            'insulation_thickness': 50.0,
            'surface_area': 8.2,
            'insulation_type': 'Silika Needeled Mat'
        },
        {
            'file_name': 'sample_4.html',
            'equipment_type': 'Compressor',
            'internal_temperature': 320.0,
            'ambient_temperature': 22.0,
            'wind_speed': 5.1,
            'insulation_thickness': 60.0,
            'surface_area': 25.8,
            'insulation_type': 'Needeled Mat'
        },
        {
            'file_name': 'sample_5.html',
            'equipment_type': 'Vertical Pipe',
            'internal_temperature': 480.0,
            'ambient_temperature': 18.0,
            'wind_speed': 3.5,
            'insulation_thickness': 80.0,
            'surface_area': 12.3,
            'insulation_type': 'Mineral Wool'
        }
    ]
    
    html_template = """
<!DOCTYPE html>
<html dir="rtl" lang="fa">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>گزارش تحلیل حرارتی - {equipment_type}</title>
    <style>
        body {{ font-family: 'Tahoma', Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 15px; text-align: center; }}
        .content {{ padding: 20px; }}
        .parameter {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #007bff; }}
        .result {{ background: #d4edda; padding: 15px; margin: 15px 0; border-radius: 5px; }}
        .value {{ font-weight: bold; color: #007bff; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 8px; text-align: right; border: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 نرم‌افزار تحلیل حرارتی کلورز (Colors)</h1>
        <p>گزارش تحلیل عایق‌کاری حرارتی</p>
    </div>
    
    <div class="content">
        <h2>📋 Project Information</h2>
        <table>
            <tr><td>Name</td><td>Sample Project</td></tr>
            <tr><td>Company</td><td>ACME Industries</td></tr>
            <tr><td>Location</td><td>Industrial Site</td></tr>
            <tr><td>Calculation Type</td><td>Steady State</td></tr>
        </table>
        
        <h2>📊 Model Summary</h2>
        <table>
            <tr><th>Name</th><th>Value</th><th>Unit</th></tr>
            <tr><td>Hot Surface Temperature</td><td>{internal_temperature}</td><td>°C</td></tr>
            <tr><td>Air Velocity</td><td>{wind_speed}</td><td>m/s</td></tr>
            <tr><td>Maximum Insulation Thickness</td><td>{insulation_thickness}</td><td>mm</td></tr>
            <tr><td>Maximum Surface Temperature</td><td>60</td><td>°C</td></tr>
        </table>
        
        <h2>🧱 Insulation Library</h2>
        <table>
            <tr><th>Index</th><th>Code</th><th>Thickness</th><th>Name</th><th>Quantity</th><th>Density</th></tr>
            <tr><td>1</td><td>RC-89</td><td>{insulation_thickness}</td><td>{insulation_type}</td><td>1</td><td>130</td></tr>
        </table>
        
        <h2>⚡ Energy and Cost Efficient Choices</h2>
        <table>
            <tr><th>Attribute</th><th>Value</th></tr>
            <tr><td>Min. Price ($/m²)</td><td>45.6</td></tr>
            <tr><td>Thickness (mm)</td><td>{insulation_thickness}</td></tr>
            <tr><td>Fin. Surface Temp (°C)</td><td>58.3</td></tr>
            <tr><td>Energy Loss (W/m²)</td><td>245.8</td></tr>
            <tr><td>Efficiency (%)</td><td>76.4</td></tr>
        </table>
        
        <div class="parameter">
            <strong>نوع تجهیز (Equipment Type):</strong> 
            <span class="value equipment_type">{equipment_type}</span>
        </div>
        
        <div class="parameter">
            <strong>دمای داخلی تجهیز (Internal Temperature):</strong> 
            <span class="value internal_temp">{internal_temperature}</span> درجه سانتیگراد
        </div>
        
        <div class="parameter">
            <strong>دمای محیط (Ambient Temperature):</strong> 
            <span class="value ambient_temp">{ambient_temperature}</span> درجه سانتیگراد
        </div>
        
        <div class="parameter">
            <strong>سرعت باد (Wind Speed):</strong> 
            <span class="value wind_speed">{wind_speed}</span> متر بر ثانیه
        </div>
        
        <div class="parameter">
            <strong>ضخامت کل عایق (Total Insulation Thickness):</strong> 
            <span class="value thickness">{insulation_thickness}</span> میلی‌متر
        </div>
        
        <div class="parameter">
            <strong>مساحت کل سطح (Total Surface Area):</strong> 
            <span class="value surface_area">{surface_area}</span> متر مربع
        </div>
        
        <div style="margin-top: 30px; text-align: center; color: #6c757d;">
            <p>تاریخ تحلیل: 2025/01/15 - 14:30</p>
            <p>نرم‌افزار کلورز - نسخه 3.0</p>
        </div>
    </div>
</body>
</html>
"""
    
    print("📝 Creating sample Colors HTML files...")
    
    for data in sample_data:
        html_content = html_template.format(**data)
        with open(data['file_name'], 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    print(f"✅ Created {len(sample_data)} sample HTML files")
    return sample_data

def demonstrate_system():
    """Demonstrate the Colors thermal analysis system"""
    
    print("🔥 Colors Thermal Analysis System - Demonstration")
    print("="*60)
    
    # Create sample HTML files
    sample_data = create_sample_colors_html()
    
    # Initialize the system
    system = ColorsThermalMLSystem()
    
    # Load Colors data
    training_data = system.load_colors_data("sample_*.html")
    
    if not training_data.empty:
        print("\n📊 Training Data Overview:")
        print(training_data.head())
        
        # Train models
        results = system.train_models()
        
        if results:
            print("\n🤖 Model Training Results:")
            for model_name, metrics in results.items():
                print(f"  • {model_name}: R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:.2f}°C")
        
        # Test predictions with various scenarios
        test_scenarios = [
            {
                'name': 'High Temperature Turbine',
                'equipment_type': 'Turbine',
                'insulation_type': 'Cerablanket',
                'insulation_thickness': 100.0,
                'surface_area': 50.0,
                'internal_temperature': 600.0,
                'ambient_temperature': 25.0,
                'wind_speed': 5.0
            },
            {
                'name': 'Medium Temperature Pipe',
                'equipment_type': 'Horizontal Pipe',
                'insulation_type': 'Mineral Wool',
                'insulation_thickness': 80.0,
                'surface_area': 20.0,
                'internal_temperature': 400.0,
                'ambient_temperature': 20.0,
                'wind_speed': 3.0
            },
            {
                'name': 'Low Temperature Valve',
                'equipment_type': 'Valve',
                'insulation_type': 'Needeled Mat',
                'insulation_thickness': 50.0,
                'surface_area': 5.0,
                'internal_temperature': 250.0,
                'ambient_temperature': 30.0,
                'wind_speed': 2.0
            }
        ]
        
        print("\n🔮 Prediction Results for Test Scenarios:")
        print("="*60)
        
        results_summary = []
        
        for scenario in test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            print("-" * 40)
            
            # Make prediction
            predictions = system.predict_surface_temperature(scenario)
            
            if predictions:
                print(f"  • Physics Prediction: {predictions['physics_prediction']:.1f}°C")
                print(f"  • ML Prediction: {predictions['ml_prediction']:.1f}°C")
                print(f"  • Combined Prediction: {predictions['combined_prediction']:.1f}°C")
                print(f"  • Thermal Conductivity: {predictions['thermal_conductivity']:.4f} W/m·K")
                
                # Calculate efficiency
                temp_reduction = scenario['internal_temperature'] - predictions['combined_prediction']
                max_temp_diff = scenario['internal_temperature'] - scenario['ambient_temperature']
                efficiency = (temp_reduction / max_temp_diff) * 100
                
                print(f"  • Insulation Efficiency: {efficiency:.1f}%")
                
                # Store results
                results_summary.append({
                    'Scenario': scenario['name'],
                    'Equipment': scenario['equipment_type'],
                    'Insulation': scenario['insulation_type'],
                    'Thickness (mm)': scenario['insulation_thickness'],
                    'Internal Temp (°C)': scenario['internal_temperature'],
                    'Predicted Surface Temp (°C)': predictions['combined_prediction'],
                    'Efficiency (%)': efficiency
                })
        
        # Create summary DataFrame
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            print("\n📈 Results Summary:")
            print(summary_df.to_string(index=False))
            
            # Save results to Excel
            summary_df.to_excel('colors_analysis_results.xlsx', index=False)
            print("\n✅ Results saved to colors_analysis_results.xlsx")
        
        # Generate detailed report for first scenario
        if test_scenarios and results_summary:
            first_scenario = test_scenarios[0]
            predictions = system.predict_surface_temperature(first_scenario)
            
            if predictions:
                report = system.generate_report(first_scenario, predictions)
                
                # Save report
                with open("colors_demo_report.txt", "w", encoding="utf-8") as f:
                    f.write(report)
                
                print(f"\n📄 Detailed report for '{first_scenario['name']}' saved to colors_demo_report.txt")
        
        # Save the trained model
        system.save_model("colors_demo_model.joblib")
        
        # Create visualization
        create_analysis_plots(results_summary)
        
    else:
        print("❌ No training data available for demonstration")
    
    # Clean up sample files
    cleanup_sample_files()

def create_analysis_plots(results_summary):
    """Create visualization plots for the analysis results"""
    
    if not results_summary:
        return
    
    print("\n📊 Creating analysis visualizations...")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    df = pd.DataFrame(results_summary)
    
    # Plot 1: Surface Temperature vs Internal Temperature
    ax1.scatter(df['Internal Temp (°C)'], df['Predicted Surface Temp (°C)'], 
                c='red', s=100, alpha=0.7)
    ax1.plot([df['Internal Temp (°C)'].min(), df['Internal Temp (°C)'].max()], 
             [df['Internal Temp (°C)'].min(), df['Internal Temp (°C)'].max()], 
             'k--', alpha=0.5, label='Perfect Insulation')
    ax1.set_xlabel('Internal Temperature (°C)')
    ax1.set_ylabel('Predicted Surface Temperature (°C)')
    ax1.set_title('Surface vs Internal Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Insulation Efficiency by Equipment Type
    equipment_types = df['Equipment'].unique()
    efficiencies = [df[df['Equipment'] == eq]['Efficiency (%)'].mean() for eq in equipment_types]
    
    bars = ax2.bar(equipment_types, efficiencies, color=['skyblue', 'lightgreen', 'salmon'])
    ax2.set_xlabel('Equipment Type')
    ax2.set_ylabel('Insulation Efficiency (%)')
    ax2.set_title('Insulation Efficiency by Equipment Type')
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, eff in zip(bars, efficiencies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{eff:.1f}%', ha='center', va='bottom')
    
    # Plot 3: Thickness vs Efficiency
    ax3.scatter(df['Thickness (mm)'], df['Efficiency (%)'], 
                c=df['Internal Temp (°C)'], s=100, alpha=0.7, cmap='coolwarm')
    ax3.set_xlabel('Insulation Thickness (mm)')
    ax3.set_ylabel('Insulation Efficiency (%)')
    ax3.set_title('Thickness vs Efficiency (colored by Internal Temp)')
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Internal Temperature (°C)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Temperature Reduction
    temp_reductions = df['Internal Temp (°C)'] - df['Predicted Surface Temp (°C)']
    
    bars = ax4.bar(range(len(df)), temp_reductions, 
                   color=['gold', 'lightcoral', 'lightblue'])
    ax4.set_xlabel('Scenario')
    ax4.set_ylabel('Temperature Reduction (°C)')
    ax4.set_title('Temperature Reduction by Scenario')
    ax4.set_xticks(range(len(df)))
    ax4.set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in df['Scenario']], 
                        rotation=45, ha='right')
    
    # Add value labels
    for i, reduction in enumerate(temp_reductions):
        ax4.text(i, reduction + 5, f'{reduction:.0f}°C', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('colors_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Analysis plots saved to colors_analysis_plots.png")

def cleanup_sample_files():
    """Clean up sample HTML files"""
    sample_files = ['sample_1.html', 'sample_2.html', 'sample_3.html', 
                   'sample_4.html', 'sample_5.html']
    
    for file in sample_files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except:
            pass
    
    print("🧹 Sample files cleaned up")

def show_insulation_database():
    """Display the insulation materials database"""
    
    print("\n🧱 Insulation Materials Database:")
    print("="*50)
    
    db = ColorsInsulationDatabase()
    
    for material, properties in db.insulation_properties.items():
        print(f"\n📋 {material}:")
        print(f"  • Available Thicknesses: {properties['thicknesses']} mm")
        print(f"  • Densities: {properties['densities']} kg/m³")
        print(f"  • Thermal Conductivity: {properties['thermal_conductivity']} W/m·K")
        print(f"  • Maximum Temperature: {properties['max_temperature']}°C")
        print(f"  • Type: {properties['type']}")

if __name__ == "__main__":
    print("🚀 Starting Colors Thermal Analysis System Demonstration")
    
    # Show insulation database
    show_insulation_database()
    
    # Run main demonstration
    demonstrate_system()
    
    print("\n🎉 Demonstration completed successfully!")
    print("📁 Generated files:")
    print("  • colors_analysis_results.xlsx - Analysis results")
    print("  • colors_demo_report.txt - Detailed report")
    print("  • colors_analysis_plots.png - Visualization plots")
    print("  • colors_demo_model.joblib - Trained ML model")