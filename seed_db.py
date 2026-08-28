
import os
import csv
import mysql.connector

# MySQL Configuration (Adjust password according to your local MySQL installation)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Ridhinya@2008',  # Put your MySQL root password here
}

CSV_FILE = os.path.join("data", "breed_information.csv")

def seed_database():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 1. Read and execute schema.sql
        with open(os.path.join("data", "schema.sql"), 'r') as f:
            schema_sql = f.read()
            
        for statement in schema_sql.split(';'):
            if statement.strip():
                cursor.execute(statement)
        
        print("✅ Database schema and tables verified/created successfully.")

        # 2. Insert CSV Breed Data
        conn.database = 'livestock_db'
        
        with open(CSV_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            insert_query = """
            INSERT INTO breed_information 
            (breed_name, type, also_known_as, native_region, coat_colour, horn_type, body_type, milk_yield_per_lactation_kg, fat_percentage, primary_use, climate_suitability, feeding_advice, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            type=VALUES(type), native_region=VALUES(native_region), milk_yield_per_lactation_kg=VALUES(milk_yield_per_lactation_kg), fat_percentage=VALUES(fat_percentage);
            """

            for row in reader:
                values = (
                    row['breed_name'], row['type'], row['also_known_as'], row['native_region'],
                    row['coat_colour'], row['horn_type'], row['body_type'],
                    row['milk_yield_per_lactation_kg'], row['fat_percentage'],
                    row['primary_use'], row['climate_suitability'], row['feeding_advice'], row['source']
                )
                cursor.execute(insert_query, values)

        conn.commit()
        print(f"✅ Successfully seeded breed information into 'livestock_db.breed_information'!")

        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"❌ MySQL Error: {err}")

if __name__ == "__main__":
    seed_database()