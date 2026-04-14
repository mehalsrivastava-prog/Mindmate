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

    prediction ENUM('Low', 'Medium', 'High') NOT NULL,
	confidence FLOAT,
    
    academic_pressure FLOAT,   
    dietary_habits FLOAT,        
    financial_stress FLOAT,      
    depression FLOAT,            

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- FOREIGN KEY
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

select * from users;
select * from checkins;
insert into users values (1,"test",'test@gmail.com','1234','2026-04-01 09:00:00');

ALTER TABLE checkins ADD confidence FLOAT;

ALTER TABLE checkins 
MODIFY depression VARCHAR(20),
MODIFY dietary_habits VARCHAR(20),
MODIFY academic_pressure VARCHAR(20);

ALTER TABLE checkins 
MODIFY prediction VARCHAR(20);
