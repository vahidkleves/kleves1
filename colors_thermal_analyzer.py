#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colors Thermal Analysis System with Machine Learning
نظام تحلیل حرارتی نرم‌افزار کلورز با یادگیری ماشین

This system parses HTML output from Colors software and uses machine learning
combined with heat transfer equations to predict surface temperatures.
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import glob
import re
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']

# Physics and mathematics
import scipy.optimize as opt
from scipy.interpolate import interp1d
import math

class ColorsInsulationDatabase:
    """Database of insulation materials with thermal properties"""
    
    def __init__(self):
        # Insulation properties database based on the specification
        self.insulation_properties = {
            'Cerablanket': {
                'thicknesses': [13, 25, 50],  # mm
                'densities': [96, 128],       # kg/m³
                'thermal_conductivity': 0.048,  # W/m·K (typical at 200°C)
                'max_temperature': 1260,      # °C
                'type': 'ceramic_fiber'
            },
            'Silika Needeled Mat': {
                'thicknesses': [3, 12],       # mm
                'densities': [150],           # kg/m³
                'thermal_conductivity': 0.042, # W/m·K
                'max_temperature': 1000,      # °C
                'type': 'silica_fiber'
            },
            'Mineral Wool': {
                'thicknesses': [25, 30, 40, 50, 70, 80, 100], # mm
                'densities': [130],           # kg/m³
                'thermal_conductivity': 0.040, # W/m·K
                'max_temperature': 750,       # °C
                'type': 'mineral_wool'
            },
            'Needeled Mat': {
                'thicknesses': [6, 10, 12, 25], # mm
                'densities': [160],           # kg/m³
                'thermal_conductivity': 0.045, # W/m·K
                'max_temperature': 850,       # °C
                'type': 'needled_mat'
            }
        }
    
    def get_insulation_property(self, material: str, property_name: str):
        """Get specific property of insulation material"""
        if material in self.insulation_properties:
            return self.insulation_properties[material].get(property_name)
        return None
    
    def get_thermal_conductivity(self, material: str, temperature: float = 200) -> float:
        """Get thermal conductivity adjusted for temperature"""
        base_k = self.get_insulation_property(material, 'thermal_conductivity')
        if base_k is None:
            return 0.05  # Default value
        
        # Temperature correction factor (simplified)
        temp_factor = 1 + 0.0002 * (temperature - 200)  # Linear approximation
        return base_k * temp_factor
    
    def validate_insulation_config(self, material: str, thickness: float, density: float) -> bool:
        """Validate if insulation configuration is available"""
        props = self.insulation_properties.get(material)
        if not props:
            return False
        
        thickness_valid = thickness in props['thicknesses']
        density_valid = density in props['densities']
        
        return thickness_valid and density_valid

class HeatTransferCalculator:
    """Physics-based heat transfer calculations"""
    
    def __init__(self):
        self.stefan_boltzmann = 5.67e-8  # W/m²·K⁴
        
    def conduction_resistance(self, thickness: float, thermal_conductivity: float, area: float) -> float:
        """Calculate thermal resistance for conduction through insulation
        
        Args:
            thickness: Insulation thickness (m)
            thermal_conductivity: k value (W/m·K)
            area: Surface area (m²)
        
        Returns:
            Thermal resistance (K/W)
        """
        return thickness / (thermal_conductivity * area)
    
    def convection_coefficient(self, wind_speed: float, characteristic_length: float = 1.0, 
                              temperature_diff: float = 100) -> float:
        """Calculate convective heat transfer coefficient
        
        Args:
            wind_speed: Wind velocity (m/s)
            characteristic_length: Characteristic length (m)
            temperature_diff: Temperature difference (K)
        
        Returns:
            Convective heat transfer coefficient (W/m²·K)
        """
        # Simplified correlation for external forced convection
        # Nu = 0.037 * Re^0.8 * Pr^(1/3) for turbulent flow
        
        # Air properties at average temperature
        rho = 1.2  # kg/m³ (air density)
        mu = 1.8e-5  # Pa·s (dynamic viscosity)
        k_air = 0.025  # W/m·K (thermal conductivity of air)
        cp = 1005  # J/kg·K (specific heat)
        pr = cp * mu / k_air  # Prandtl number
        
        # Reynolds number
        re = rho * wind_speed * characteristic_length / mu
        
        # Nusselt number
        if re > 5e5:  # Turbulent
            nu = 0.037 * (re**0.8) * (pr**(1/3))
        else:  # Laminar
            nu = 0.664 * (re**0.5) * (pr**(1/3))
        
        # Convective heat transfer coefficient
        h = nu * k_air / characteristic_length
        
        # Minimum value for natural convection
        h_natural = 5 + 3.8 * wind_speed  # Simplified correlation
        
        return max(h, h_natural)
    
    def radiation_heat_transfer(self, surface_temp: float, ambient_temp: float, 
                               emissivity: float = 0.8) -> float:
        """Calculate radiative heat transfer
        
        Args:
            surface_temp: Surface temperature (K)
            ambient_temp: Ambient temperature (K)
            emissivity: Surface emissivity
        
        Returns:
            Radiative heat flux (W/m²)
        """
        return emissivity * self.stefan_boltzmann * (surface_temp**4 - ambient_temp**4)
    
    def calculate_surface_temperature(self, internal_temp: float, ambient_temp: float,
                                    insulation_thickness: float, thermal_conductivity: float,
                                    wind_speed: float, surface_area: float) -> float:
        """Calculate surface temperature using heat transfer equations
        
        Args:
            internal_temp: Internal equipment temperature (°C)
            ambient_temp: Ambient temperature (°C)
            insulation_thickness: Total insulation thickness (mm)
            thermal_conductivity: Effective thermal conductivity (W/m·K)
            wind_speed: Wind speed (m/s)
            surface_area: Surface area (m²)
        
        Returns:
            Surface temperature (°C)
        """
        # Convert to Kelvin and SI units
        T_internal = internal_temp + 273.15
        T_ambient = ambient_temp + 273.15
        thickness = insulation_thickness / 1000  # mm to m
        
        # Calculate characteristic length
        char_length = (surface_area / math.pi) ** 0.5  # Equivalent radius
        
        def heat_balance_equation(T_surface):
            """Heat balance equation to solve"""
            T_surf = T_surface[0] if isinstance(T_surface, np.ndarray) else T_surface
            
            # Conduction through insulation
            q_cond = (T_internal - T_surf) / self.conduction_resistance(thickness, thermal_conductivity, surface_area)
            
            # Convection to ambient
            h_conv = self.convection_coefficient(wind_speed, char_length, T_surf - T_ambient)
            q_conv = h_conv * surface_area * (T_surf - T_ambient)
            
            # Radiation to ambient
            q_rad = self.radiation_heat_transfer(T_surf, T_ambient) * surface_area
            
            # Heat balance: q_in = q_out
            return q_cond - q_conv - q_rad
        
        # Solve for surface temperature
        try:
            # Initial guess
            T_surface_guess = (T_internal + T_ambient) / 2
            
            # Solve the heat balance equation
            result = opt.fsolve(heat_balance_equation, T_surface_guess)
            T_surface = result[0]
            
            # Convert back to Celsius
            return T_surface - 273.15
            
        except:
            # Fallback to simplified calculation
            R_total = thickness / (thermal_conductivity * surface_area)
            h_avg = self.convection_coefficient(wind_speed, char_length)
            R_conv = 1 / (h_avg * surface_area)
            
            T_surface = T_internal - (T_internal - T_ambient) * R_conv / (R_total + R_conv)
            return T_surface - 273.15

class ColorsHTMLParser:
    """Parser for Colors software HTML output files"""
    
    def __init__(self):
        self.insulation_db = ColorsInsulationDatabase()
        
    def parse_html_file(self, file_path: str) -> Optional[Dict]:
        """Parse a single Colors HTML file
        
        Args:
            file_path: Path to HTML file
            
        Returns:
            Dictionary with parsed data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract project information
            project_info = self._extract_project_info(soup)
            
            # Extract model summary
            model_summary = self._extract_model_summary(soup)
            
            # Extract insulation library data
            insulation_data = self._extract_insulation_library(soup)
            
            # Extract energy and cost data
            energy_data = self._extract_energy_data(soup)
            
            # Combine all data
            data = {
                'file_name': os.path.basename(file_path),
                **project_info,
                **model_summary,
                **energy_data
            }
            
            # Add insulation details if available
            if insulation_data:
                data['insulation_layers'] = insulation_data
            
            return data
            
        except Exception as e:
            print(f"❌ Error parsing {file_path}: {str(e)}")
            return None
    
    def _extract_project_info(self, soup: BeautifulSoup) -> Dict:
        """Extract project information from HTML"""
        project_info = {}
        
        # Look for project information table
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    key = cells[0].get_text().strip()
                    value = cells[1].get_text().strip()
                    
                    # Map common fields
                    if 'name' in key.lower() or 'نام' in key:
                        project_info['project_name'] = value
                    elif 'company' in key.lower() or 'شرکت' in key:
                        project_info['company'] = value
                    elif 'location' in key.lower() or 'موقعیت' in key:
                        project_info['location'] = value
                    elif 'calculation type' in key.lower() or 'نوع محاسبه' in key:
                        project_info['calculation_type'] = value
        
        return project_info
    
    def _extract_model_summary(self, soup: BeautifulSoup) -> Dict:
        """Extract model summary data"""
        model_data = {}
        
        # Look for model summary table
        summary_section = soup.find(text=re.compile(r'Model Summary|خلاصه مدل'))
        if summary_section:
            table = summary_section.find_parent().find_next('table')
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:  # Usually has Name, Value, Unit columns
                        name = cells[0].get_text().strip()
                        value = cells[1].get_text().strip()
                        unit = cells[2].get_text().strip() if len(cells) > 2 else ''
                        
                        # Parse numeric values
                        numeric_value = self._parse_numeric(value)
                        if numeric_value is not None:
                            model_data[self._normalize_key(name)] = numeric_value
        
        return model_data
    
    def _extract_insulation_library(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract insulation library data"""
        insulation_layers = []
        
        # Look for insulation library section
        insulation_section = soup.find(text=re.compile(r'Insulation Library|کتابخانه عایق'))
        if insulation_section:
            table = insulation_section.find_parent().find_next('table')
            if table:
                rows = table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 6:  # Based on the image structure
                        layer_data = {
                            'index': self._parse_numeric(cells[0].get_text().strip()),
                            'code': cells[1].get_text().strip(),
                            'thickness': self._parse_numeric(cells[2].get_text().strip()),
                            'name': cells[3].get_text().strip(),
                            'quantity': self._parse_numeric(cells[4].get_text().strip()),
                            'density': self._parse_numeric(cells[5].get_text().strip()) if len(cells) > 5 else None,
                            'thermal_conductivity': self._parse_numeric(cells[6].get_text().strip()) if len(cells) > 6 else None,
                            'price': self._parse_numeric(cells[7].get_text().strip()) if len(cells) > 7 else None,
                            'cost': self._parse_numeric(cells[8].get_text().strip()) if len(cells) > 8 else None
                        }
                        insulation_layers.append(layer_data)
        
        return insulation_layers
    
    def _extract_energy_data(self, soup: BeautifulSoup) -> Dict:
        """Extract energy and cost efficiency data"""
        energy_data = {}
        
        # Look for energy section
        energy_section = soup.find(text=re.compile(r'Energy.*Cost.*Efficient|انرژی.*هزینه'))
        if energy_section:
            table = energy_section.find_parent().find_next('table')
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        key = cells[0].get_text().strip()
                        value = self._parse_numeric(cells[1].get_text().strip())
                        
                        if value is not None:
                            energy_data[self._normalize_key(key)] = value
        
        return energy_data
    
    def _parse_numeric(self, text: str) -> Optional[float]:
        """Parse numeric value from text"""
        try:
            # Remove Persian digits and convert to English
            persian_digits = '۰۱۲۳۴۵۶۷۸۹'
            english_digits = '0123456789'
            
            for persian, english in zip(persian_digits, english_digits):
                text = text.replace(persian, english)
            
            # Extract number using regex
            numbers = re.findall(r'[-+]?\d*\.?\d+', text)
            if numbers:
                return float(numbers[0])
            return None
        except:
            return None
    
    def _normalize_key(self, key: str) -> str:
        """Normalize key name for consistent data structure"""
        key = key.lower().strip()
        key = re.sub(r'[^\w\s]', '', key)  # Remove special characters
        key = re.sub(r'\s+', '_', key)      # Replace spaces with underscores
        return key

class ColorsThermalMLSystem:
    """Machine Learning system for thermal analysis using Colors data"""
    
    def __init__(self):
        self.html_parser = ColorsHTMLParser()
        self.heat_calculator = HeatTransferCalculator()
        self.insulation_db = ColorsInsulationDatabase()
        
        # ML components
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # Data storage
        self.training_data = pd.DataFrame()
        self.feature_columns = []
        self.target_column = 'surface_temperature'
        
    def load_colors_data(self, html_pattern: str = "*.html") -> pd.DataFrame:
        """Load and parse Colors HTML files
        
        Args:
            html_pattern: Pattern to match HTML files
            
        Returns:
            DataFrame with parsed data
        """
        print("📁 Loading Colors software data...")
        
        html_files = glob.glob(html_pattern)
        if not html_files:
            print(f"⚠️  No HTML files found matching pattern: {html_pattern}")
            return pd.DataFrame()
        
        all_data = []
        
        for file_path in html_files:
            data = self.html_parser.parse_html_file(file_path)
            if data:
                all_data.append(data)
        
        if all_data:
            self.training_data = pd.DataFrame(all_data)
            print(f"✅ Successfully loaded {len(all_data)} files")
            return self.training_data
        else:
            print("❌ No data extracted from HTML files")
            return pd.DataFrame()
    
    def get_user_input(self) -> Dict:
        """Get equipment parameters from user input"""
        print("\n🔧 لطفاً مشخصات تجهیز را وارد کنید:")
        print("Please enter equipment specifications:")
        
        user_data = {}
        
        # Equipment type
        print("\nانواع تجهیزات موجود / Available equipment types:")
        equipment_types = [
            "Horizontal Pipe", "Vertical Pipe", "Flat Surface", 
            "Sphere", "Turbine", "Valve", "Complex Equipment"
        ]
        for i, eq_type in enumerate(equipment_types, 1):
            print(f"{i}. {eq_type}")
        
        try:
            eq_choice = int(input("انتخاب نوع تجهیز (شماره) / Choose equipment type (number): ")) - 1
            user_data['equipment_type'] = equipment_types[eq_choice]
        except:
            user_data['equipment_type'] = "Complex Equipment"
        
        # Insulation type
        print("\nانواع عایق موجود / Available insulation types:")
        insulation_types = list(self.insulation_db.insulation_properties.keys())
        for i, ins_type in enumerate(insulation_types, 1):
            print(f"{i}. {ins_type}")
        
        try:
            ins_choice = int(input("انتخاب نوع عایق (شماره) / Choose insulation type (number): ")) - 1
            user_data['insulation_type'] = insulation_types[ins_choice]
        except:
            user_data['insulation_type'] = "Mineral Wool"
        
        # Get available thicknesses for selected insulation
        available_thicknesses = self.insulation_db.get_insulation_property(user_data['insulation_type'], 'thicknesses')
        print(f"\nضخامت‌های موجود / Available thicknesses: {available_thicknesses} mm")
        
        try:
            user_data['insulation_thickness'] = float(input("ضخامت عایق (mm) / Insulation thickness (mm): "))
        except:
            user_data['insulation_thickness'] = available_thicknesses[0] if available_thicknesses else 50
        
        # Other parameters
        try:
            user_data['surface_area'] = float(input("مساحت سطح (m²) / Surface area (m²): "))
        except:
            user_data['surface_area'] = 10.0
        
        try:
            user_data['internal_temperature'] = float(input("دمای داخلی تجهیز (°C) / Internal temperature (°C): "))
        except:
            user_data['internal_temperature'] = 400.0
        
        try:
            user_data['ambient_temperature'] = float(input("دمای محیط (°C) / Ambient temperature (°C): "))
        except:
            user_data['ambient_temperature'] = 25.0
        
        try:
            user_data['wind_speed'] = float(input("سرعت باد (m/s) / Wind speed (m/s): "))
        except:
            user_data['wind_speed'] = 3.0
        
        return user_data
    
    def prepare_features(self, data: Union[pd.DataFrame, Dict]) -> np.ndarray:
        """Prepare features for ML model"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Define the feature columns in consistent order
        if not self.feature_columns:
            self.feature_columns = [
                'equipment_type_encoded', 'insulation_type_encoded',
                'insulation_thickness', 'surface_area', 'internal_temperature',
                'ambient_temperature', 'wind_speed'
            ]
        
        n_samples = len(data)
        feature_matrix = []
        
        # Encode categorical variables
        for col in ['equipment_type', 'insulation_type']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                # Fit with common values
                common_values = {
                    'equipment_type': ["Horizontal Pipe", "Vertical Pipe", "Flat Surface", "Sphere", "Turbine", "Valve", "Complex Equipment"],
                    'insulation_type': list(self.insulation_db.insulation_properties.keys())
                }
                self.label_encoders[col].fit(common_values[col])
            
            # Get values for this column
            if col in data.columns:
                values = data[col].tolist()
            else:
                default_val = 'Complex Equipment' if col == 'equipment_type' else 'Mineral Wool'
                values = [default_val] * n_samples
            
            encoded_values = []
            for val in values:
                try:
                    encoded_values.append(self.label_encoders[col].transform([val])[0])
                except:
                    encoded_values.append(0)  # Default encoding
            
            feature_matrix.append(encoded_values)
        
        # Add numerical features in consistent order
        numerical_features = [
            'insulation_thickness', 'surface_area', 'internal_temperature',
            'ambient_temperature', 'wind_speed'
        ]
        
        default_values = {
            'insulation_thickness': 50.0,
            'surface_area': 10.0,
            'internal_temperature': 400.0,
            'ambient_temperature': 25.0,
            'wind_speed': 3.0
        }
        
        for feature in numerical_features:
            if feature in data.columns:
                feature_matrix.append(data[feature].tolist())
            else:
                feature_matrix.append([default_values[feature]] * n_samples)
        
        # Convert to numpy array and transpose
        features_array = np.array(feature_matrix).T
        
        return features_array
    
    def train_models(self) -> Dict:
        """Train multiple ML models and select the best one"""
        if self.training_data.empty:
            print("❌ No training data available")
            return {}
        
        print("🤖 Training machine learning models...")
        
        # Prepare features and target
        X = self.prepare_features(self.training_data)
        
        # Create target variable using physics calculations if not available
        if self.target_column not in self.training_data.columns:
            print("📊 Calculating surface temperatures using physics equations...")
            surface_temps = []
            
            for i, row in self.training_data.iterrows():
                try:
                    # Get thermal conductivity
                    ins_type = row.get('insulation_type', 'Mineral Wool')
                    k = self.insulation_db.get_thermal_conductivity(ins_type, row.get('internal_temperature', 200))
                    
                    # Calculate surface temperature
                    surface_temp = self.heat_calculator.calculate_surface_temperature(
                        internal_temp=row.get('internal_temperature', 400),
                        ambient_temp=row.get('ambient_temperature', 25),
                        insulation_thickness=row.get('insulation_thickness', 50),
                        thermal_conductivity=k,
                        wind_speed=row.get('wind_speed', 3),
                        surface_area=row.get('surface_area', 10)
                    )
                    surface_temps.append(surface_temp)
                except:
                    surface_temps.append(100)  # Default value
            
            self.training_data[self.target_column] = surface_temps
        
        y = self.training_data[self.target_column].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train and evaluate models
        results = {}
        
        for name, model in self.models.items():
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
                
                # Train on full dataset
                model.fit(X_scaled, y)
                
                # Predictions
                y_pred = model.predict(X_scaled)
                
                # Metrics
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                
                results[name] = {
                    'model': model,
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std(),
                    'mse': mse,
                    'r2': r2,
                    'mae': mae
                }
                
                print(f"✅ {name}: R² = {r2:.3f}, MAE = {mae:.2f}°C")
                
            except Exception as e:
                print(f"❌ Error training {name}: {str(e)}")
        
        # Select best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
            self.best_model = results[best_model_name]['model']
            print(f"🏆 Best model: {best_model_name}")
        
        return results
    
    def predict_surface_temperature(self, user_input: Dict) -> Dict:
        """Predict surface temperature for user input"""
        if self.best_model is None:
            print("❌ No trained model available")
            return {}
        
        print("🔮 Predicting surface temperature...")
        
        # Physics-based prediction
        ins_type = user_input.get('insulation_type', 'Mineral Wool')
        k = self.insulation_db.get_thermal_conductivity(ins_type, user_input.get('internal_temperature', 200))
        
        physics_temp = self.heat_calculator.calculate_surface_temperature(
            internal_temp=user_input['internal_temperature'],
            ambient_temp=user_input['ambient_temperature'],
            insulation_thickness=user_input['insulation_thickness'],
            thermal_conductivity=k,
            wind_speed=user_input['wind_speed'],
            surface_area=user_input['surface_area']
        )
        
        # ML-based prediction
        X_user = self.prepare_features(user_input)
        X_user_scaled = self.scaler.transform(X_user)
        ml_temp = self.best_model.predict(X_user_scaled)[0]
        
        # Combined prediction (weighted average)
        combined_temp = 0.6 * ml_temp + 0.4 * physics_temp
        
        return {
            'physics_prediction': physics_temp,
            'ml_prediction': ml_temp,
            'combined_prediction': combined_temp,
            'thermal_conductivity': k,
            'insulation_type': ins_type
        }
    
    def generate_report(self, user_input: Dict, predictions: Dict) -> str:
        """Generate detailed analysis report"""
        report = f"""
🔥 گزارش تحلیل حرارتی سیستم کلورز
Colors Thermal Analysis Report
{'='*50}

📋 مشخصات تجهیز / Equipment Specifications:
• نوع تجهیز / Equipment Type: {user_input['equipment_type']}
• نوع عایق / Insulation Type: {user_input['insulation_type']}
• ضخامت عایق / Insulation Thickness: {user_input['insulation_thickness']} mm
• مساحت سطح / Surface Area: {user_input['surface_area']} m²
• دمای داخلی / Internal Temperature: {user_input['internal_temperature']}°C
• دمای محیط / Ambient Temperature: {user_input['ambient_temperature']}°C
• سرعت باد / Wind Speed: {user_input['wind_speed']} m/s

🌡️ نتایج پیش‌بینی / Prediction Results:
• پیش‌بینی فیزیکی / Physics Prediction: {predictions['physics_prediction']:.1f}°C
• پیش‌بینی یادگیری ماشین / ML Prediction: {predictions['ml_prediction']:.1f}°C
• پیش‌بینی ترکیبی / Combined Prediction: {predictions['combined_prediction']:.1f}°C

🔬 خصوصیات حرارتی / Thermal Properties:
• ضریب هدایت حرارتی / Thermal Conductivity: {predictions['thermal_conductivity']:.4f} W/m·K
• اختلاف دما / Temperature Difference: {user_input['internal_temperature'] - user_input['ambient_temperature']:.1f}°C

📊 تحلیل عملکرد / Performance Analysis:
• کیفیت عایق‌کاری / Insulation Quality: {'عالی' if predictions['combined_prediction'] < user_input['ambient_temperature'] + 50 else 'متوسط' if predictions['combined_prediction'] < user_input['ambient_temperature'] + 100 else 'ضعیف'}
• بازدهی انرژی / Energy Efficiency: {((user_input['internal_temperature'] - predictions['combined_prediction']) / (user_input['internal_temperature'] - user_input['ambient_temperature']) * 100):.1f}%

⚠️  توصیه‌ها / Recommendations:
"""
        
        # Add recommendations
        temp_diff = predictions['combined_prediction'] - user_input['ambient_temperature']
        if temp_diff > 100:
            report += "• افزایش ضخامت عایق یا استفاده از عایق بهتر توصیه می‌شود\n"
            report += "• Increase insulation thickness or use better insulation material\n"
        elif temp_diff < 30:
            report += "• عایق‌کاری بهینه است\n"
            report += "• Insulation is optimal\n"
        else:
            report += "• عایق‌کاری قابل قبول است\n"
            report += "• Insulation is acceptable\n"
        
        return report
    
    def save_model(self, filename: str = "colors_thermal_model.joblib"):
        """Save trained model"""
        if self.best_model is not None:
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns
            }
            joblib.dump(model_data, filename)
            print(f"✅ Model saved to {filename}")
    
    def load_model(self, filename: str = "colors_thermal_model.joblib"):
        """Load trained model"""
        try:
            model_data = joblib.load(filename)
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data.get('feature_columns', [])
            print(f"✅ Model loaded from {filename}")
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")

def main():
    """Main function to run the Colors thermal analysis system"""
    print("🔥 سیستم تحلیل حرارتی نرم‌افزار کلورز")
    print("Colors Thermal Analysis System")
    print("="*50)
    
    # Initialize system
    system = ColorsThermalMLSystem()
    
    # Load Colors data
    system.load_colors_data("*.html")
    
    # Train models
    system.train_models()
    
    # Get user input
    user_input = system.get_user_input()
    
    # Make predictions
    predictions = system.predict_surface_temperature(user_input)
    
    # Generate and display report
    if predictions:
        report = system.generate_report(user_input, predictions)
        print(report)
        
        # Save model
        system.save_model()
        
        # Save report to file
        with open("thermal_analysis_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("\n✅ گزارش در فایل thermal_analysis_report.txt ذخیره شد")
        print("✅ Report saved to thermal_analysis_report.txt")
    
    else:
        print("❌ خطا در پیش‌بینی / Error in prediction")

if __name__ == "__main__":
    main()