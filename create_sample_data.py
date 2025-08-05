#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample Data Generator for Thermal Insulation Analysis System
Creates sample HTML files (Kloriz output) and Excel file (insulation data)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_sample_html_files(num_files=10):
    """Create sample HTML files simulating Kloriz software output"""
    
    print("🔧 Creating sample HTML files...")
    
    # Equipment types and their typical parameters
    equipment_data = {
        'Horizontal Pipe': {'area_range': (1, 5), 'complexity': 1},
        'Vertical Pipe': {'area_range': (1, 5), 'complexity': 1},
        'Horizontal Flat Surface': {'area_range': (5, 20), 'complexity': 1.2},
        'Vertical Flat Surface': {'area_range': (5, 20), 'complexity': 1.2},
        'Sphere': {'area_range': (2, 8), 'complexity': 1.1},
        'Cube': {'area_range': (3, 12), 'complexity': 1.3},
        'Turbine V94.2': {'area_range': (20, 50), 'complexity': 1.8},
        'Valve': {'area_range': (0.5, 3), 'complexity': 1.4},
        'Compressor': {'area_range': (10, 30), 'complexity': 1.6},
        'Complex Turbine': {'area_range': (25, 60), 'complexity': 2.0}
    }
    
    np.random.seed(42)
    
    for i in range(1, num_files + 1):
        equipment_type = np.random.choice(list(equipment_data.keys()))
        equipment_props = equipment_data[equipment_type]
        
        # Generate realistic parameters
        internal_temp = np.random.uniform(200, 650)  # °C
        ambient_temp = np.random.uniform(15, 45)     # °C
        wind_speed = np.random.uniform(0.5, 8.0)     # m/s
        surface_area = np.random.uniform(*equipment_props['area_range'])  # m²
        thickness = np.random.uniform(50, 150)       # mm
        
        # Calculate surface temperature using simplified heat transfer
        delta_t = internal_temp - ambient_temp
        heat_loss_factor = np.exp(-thickness/100) * equipment_props['complexity']
        surface_temp = ambient_temp + delta_t * heat_loss_factor * (1 + wind_speed/20)
        surface_temp += np.random.normal(0, 3)  # Add some noise
        surface_temp = max(ambient_temp + 5, surface_temp)  # Ensure it's above ambient
        
        # Create HTML content in Persian/English
        html_content = f"""
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
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 نرم‌افزار تحلیل حرارتی کلورز (Kloriz)</h1>
        <p>گزارش تحلیل عایق‌کاری حرارتی</p>
    </div>
    
    <div class="content">
        <h2>📋 مشخصات تجهیز</h2>
        
        <div class="parameter">
            <strong>نوع تجهیز (Equipment Type):</strong> 
            <span class="value equipment_type">{equipment_type}</span>
        </div>
        
        <div class="parameter">
            <strong>دمای داخلی تجهیز (Internal Temperature):</strong> 
            <span class="value internal_temp">{internal_temp:.1f}</span> درجه سانتیگراد
        </div>
        
        <div class="parameter">
            <strong>دمای محیط (Ambient Temperature):</strong> 
            <span class="value ambient_temp">{ambient_temp:.1f}</span> درجه سانتیگراد
        </div>
        
        <div class="parameter">
            <strong>سرعت باد (Wind Speed):</strong> 
            <span class="value wind_speed">{wind_speed:.1f}</span> متر بر ثانیه
        </div>
        
        <div class="parameter">
            <strong>ضخامت کل عایق (Total Insulation Thickness):</strong> 
            <span class="value thickness">{thickness:.1f}</span> میلی‌متر
        </div>
        
        <div class="parameter">
            <strong>مساحت کل سطح (Total Surface Area):</strong> 
            <span class="value surface_area">{surface_area:.2f}</span> متر مربع
        </div>
        
        <div class="result">
            <h3>🌡️ نتیجه تحلیل</h3>
            <p><strong>دمای سطح خارجی عایق (Surface Temperature):</strong> 
            <span class="value surface_temp">{surface_temp:.1f}</span> درجه سانتیگراد</p>
        </div>
        
        <div style="margin-top: 30px; text-align: center; color: #6c757d;">
            <p>تاریخ تحلیل: {datetime.now().strftime('%Y/%m/%d - %H:%M')}</p>
            <p>نرم‌افزار کلورز - نسخه 2.1</p>
        </div>
    </div>
</body>
</html>
        """
        
        filename = f"/workspace/a{i}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Created: {filename}")
    
    print(f"🎉 Successfully created {num_files} sample HTML files")

def create_sample_excel_file():
    """Create sample Excel file with insulation properties"""
    
    print("📊 Creating sample Excel file...")
    
    np.random.seed(42)
    
    # Define insulation types and their properties
    insulation_types = {
        'Cerablanket': {
            'density_range': (120, 135),
            'thermal_conductivity_range': (0.040, 0.050),
            'max_temp': 1260,
            'description': 'عایق سرامیکی با مقاومت حرارتی بالا'
        },
        'Silika Needeled Mat': {
            'density_range': (90, 105),
            'thermal_conductivity_range': (0.035, 0.042),
            'max_temp': 1000,
            'description': 'پشم سیلیکا سوزنی شده'
        },
        'Rock Wool': {
            'density_range': (130, 150),
            'thermal_conductivity_range': (0.038, 0.046),
            'max_temp': 750,
            'description': 'پشم سنگ معدنی'
        },
        'Needeled Mat': {
            'density_range': (110, 130),
            'thermal_conductivity_range': (0.037, 0.043),
            'max_temp': 650,
            'description': 'پشم سوزنی شده'
        }
    }
    
    insulation_data = []
    
    # Generate sample data for each insulation type
    for i in range(100):
        insulation_type = np.random.choice(list(insulation_types.keys()))
        props = insulation_types[insulation_type]
        
        insulation_data.append({
            'Layer_ID': i + 1,
            'Insulation_Type': insulation_type,
            'Description': props['description'],
            'Thickness_mm': np.random.uniform(20, 120),
            'Density_kg_m3': np.random.uniform(*props['density_range']),
            'Thermal_Conductivity_W_mK': np.random.uniform(*props['thermal_conductivity_range']),
            'Convection_Coefficient_W_m2K': np.random.uniform(8, 25),
            'Max_Temperature_C': props['max_temp'],
            'Cost_per_m2_USD': np.random.uniform(15, 45),
            'Installation_Difficulty': np.random.choice(['آسان', 'متوسط', 'سخت']),
            'Fire_Resistance': np.random.choice(['بالا', 'متوسط', 'پایین'])
        })
    
    # Create DataFrame
    df = pd.DataFrame(insulation_data)
    
    # Save to Excel file
    excel_file = '/workspace/inputdata.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Insulation_Data', index=False)
        
        # Create a summary sheet
        summary_data = []
        for insulation_type, props in insulation_types.items():
            type_data = df[df['Insulation_Type'] == insulation_type]
            summary_data.append({
                'Insulation_Type': insulation_type,
                'Description': props['description'],
                'Count': len(type_data),
                'Avg_Density': type_data['Density_kg_m3'].mean(),
                'Avg_Thermal_Conductivity': type_data['Thermal_Conductivity_W_mK'].mean(),
                'Max_Temperature': props['max_temp'],
                'Avg_Cost': type_data['Cost_per_m2_USD'].mean()
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✅ Created: {excel_file}")
    print(f"📊 Data shape: {df.shape}")
    print(f"📋 Insulation types: {df['Insulation_Type'].nunique()}")
    
    return df

def main():
    """Main function to create all sample data"""
    
    print("🚀 Sample Data Generator for Thermal Insulation Analysis")
    print("=" * 60)
    
    # Create sample HTML files
    create_sample_html_files(15)  # Create 15 sample files
    
    print()
    
    # Create sample Excel file
    create_sample_excel_file()
    
    print()
    print("🎉 All sample data created successfully!")
    print("📁 Files created:")
    print("   - a1.html to a15.html (Kloriz output files)")
    print("   - inputdata.xlsx (Insulation properties)")
    print()
    print("🔧 Next steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Run the main analysis: python thermal_insulation_analyzer.py")

if __name__ == "__main__":
    main()