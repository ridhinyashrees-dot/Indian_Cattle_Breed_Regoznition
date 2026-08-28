-- Create Database
CREATE DATABASE IF NOT EXISTS livestock_db;
USE livestock_db;

-- Table 1: Breed Information Database
CREATE TABLE IF NOT EXISTS breed_information (
    breed_id INT AUTO_INCREMENT PRIMARY KEY,
    breed_name VARCHAR(100) NOT NULL UNIQUE,
    type VARCHAR(50) NOT NULL,
    also_known_as VARCHAR(150),
    native_region VARCHAR(150),
    coat_colour VARCHAR(150),
    horn_type VARCHAR(150),
    body_type VARCHAR(150),
    milk_yield_per_lactation_kg VARCHAR(50),
    fat_percentage VARCHAR(50),
    primary_use VARCHAR(100),
    climate_suitability VARCHAR(150),
    feeding_advice TEXT,
    source VARCHAR(100)
);

-- Table 2: Vaccination Advisory Information
CREATE TABLE IF NOT EXISTS vaccination_information (
    vaccination_id INT AUTO_INCREMENT PRIMARY KEY,
    animal_type VARCHAR(50) NOT NULL,
    disease VARCHAR(100) NOT NULL,
    vaccine VARCHAR(100) NOT NULL,
    recommended_age VARCHAR(100),
    frequency VARCHAR(100),
    notes TEXT,
    source VARCHAR(100)
);

-- Table 3: Digital Animal Registration
CREATE TABLE IF NOT EXISTS animal_registration (
    animal_id INT AUTO_INCREMENT PRIMARY KEY,
    registration_tag VARCHAR(50) UNIQUE NOT NULL,
    owner_name VARCHAR(100) NOT NULL,
    mobile_number VARCHAR(15) NOT NULL,
    village VARCHAR(100) NOT NULL,
    animal_type VARCHAR(50) NOT NULL,
    predicted_breed VARCHAR(100) NOT NULL,
    prediction_confidence FLOAT NOT NULL,
    registration_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_path VARCHAR(255),
    verification_status VARCHAR(50) DEFAULT 'Pending Verification'
);

-- Seed Vaccination Information
INSERT INTO vaccination_information (animal_type, disease, vaccine, recommended_age, frequency, notes, source) VALUES
('Buffalo', 'Foot and Mouth Disease (FMD)', 'FMD Polyvalent Vaccine', '4 months and above', 'Bi-annually (Every 6 months)', 'Consult local veterinary surgeon before vaccination.', 'NDDB Guidelines'),
('Buffalo', 'Hemorrhagic Septicemia (HS)', 'HS Oil Adjuvant Vaccine', '6 months and above', 'Annually (Pre-monsoon)', 'Critical before rainy season in humid endemic regions.', 'DADF Govt of India'),
('Buffalo', 'Black Quarter (BQ)', 'BQ Vaccine', '6 months and above', 'Annually (Pre-monsoon)', 'Endemic areas require annual booster.', 'DADF Govt of India'),
('Buffalo', 'Brucellosis', 'Brucella abortus S19 Vaccine', '4-8 months (Female calves only)', 'Once in a lifetime', 'Do not administer to adult pregnant animals.', 'ICAR-NIVEDI');