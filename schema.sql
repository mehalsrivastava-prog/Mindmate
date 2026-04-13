CREATE DATABASE mindmate_db;

USE mindmate_db;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE checkins (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,

    sleep FLOAT,
    work_hours INT,
    activity FLOAT,
    social INT,
    stress_self INT,

    prediction ENUM('Low', 'Medium', 'High') NOT NULL,
    confidence INT,

    academic_pressure FLOAT,     
    study_satisfaction FLOAT,    
    dietary_habits FLOAT,        
    financial_stress FLOAT,      
    depression FLOAT,            

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- FOREIGN KEY
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);


