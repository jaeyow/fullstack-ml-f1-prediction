# pylint: disable=C0415,C0103,W0201,W0702,W0718
import os
from metaflow import FlowSpec, step
# from comet_ml.integration.metaflow import comet_flow

try:
    from dotenv import load_dotenv

    load_dotenv(verbose=True, dotenv_path=".env")
except Exception:
    print("No dotenv package")


# @comet_flow(project_name="<please-replace-me>")
class F1PredictionFeaturePipeline(FlowSpec):

    @step
    def start(self):
        import duckdb
        """
        Initialization, place everything init related here, check that everything is
        in order like environment variables, connection strings, etc, and if there are
        any issues, quit now.
        """
        # assert os.environ["AWS_PROFILE_NAME"]
        # assert os.environ["AWS_DEFAULT_REGION"]
        # assert os.environ["BUCKET_NAME"]
        # assert os.environ["COMET_API_KEY"]

        # self.AWS_PROFILE_NAME = os.environ["AWS_PROFILE_NAME"]
        # self.AWS_DEFAULT_REGION = os.environ["AWS_DEFAULT_REGION"]
        # self.BUCKET_NAME = os.environ["BUCKET_NAME"]
        print("F1PredictionFeaturePipeline...")
        
        self.next(self.get_dataset)

    @step
    def get_dataset(self):
        """
        get_dataset
        """
        import duckdb
        import numpy as np
        import pandas as pd

        print("get_dataset...")
        con = duckdb.connect(database = "f1-race-data-2023.duckdb", read_only = False)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS circuits AS SELECT * FROM '../data/circuits.csv';
            CREATE TABLE IF NOT EXISTS constructor_results AS SELECT * FROM '../data/constructor_results.csv';
            CREATE TABLE IF NOT EXISTS constructor_standings AS SELECT * FROM '../data/constructor_standings.csv';
            CREATE TABLE IF NOT EXISTS constructors AS SELECT * FROM '../data/constructors.csv';
            CREATE TABLE IF NOT EXISTS driver_standings AS SELECT * FROM '../data/driver_standings.csv';
            CREATE TABLE IF NOT EXISTS drivers AS SELECT * FROM '../data/drivers.csv';
            CREATE TABLE IF NOT EXISTS lap_times AS SELECT * FROM '../data/lap_times.csv';
            CREATE TABLE IF NOT EXISTS pit_stops AS SELECT * FROM '../data/pit_stops.csv';
            CREATE TABLE IF NOT EXISTS qualifying AS SELECT * FROM '../data/qualifying.csv';
            CREATE TABLE IF NOT EXISTS races AS SELECT * FROM '../data/races.csv';
            CREATE TABLE IF NOT EXISTS results AS SELECT * FROM '../data/results.csv';
            CREATE TABLE IF NOT EXISTS seasons AS SELECT * FROM '../data/seasons.csv';
            CREATE TABLE IF NOT EXISTS sprint_results AS SELECT * FROM '../data/sprint_results.csv';
            CREATE TABLE IF NOT EXISTS status AS SELECT * FROM '../data/status.csv';
            CREATE TABLE IF NOT EXISTS results_store AS SELECT * FROM results;
            """
        )
        
        # check if tables are created
        drv_df = con.execute(
            """
            SELECT
                drivers.driverId,
                drivers.surname,
                drivers.forename,
                standings.points,
                races.name,
                races.year
            FROM '../data/drivers.csv' as drivers
            INNER JOIN '../data/driver_standings.csv' as standings
                ON drivers.driverId = standings.driverId
            INNER JOIN '../data/races.csv' as races
                ON standings.raceId = races.raceId
            WHERE surname = 'Ricciardo' AND standings.position <= 5
            ORDER BY standings.points DESC;
            """
        ).df()
        print(drv_df.head(5))

        results_store = con.execute(
            """
            SELECT * FROM results_store;
            """
        ).df()
        print(results_store.head(5))
        
        self.next(self.create_driver_experience_feature)

    @step
    def create_driver_experience_feature(self):
        """
        Feature Engineering: Driver's experience in Formula 1, where a more experienced F1 driver
        typically places better than a rookie. 
        """
        import duckdb
        import numpy as np
        import pandas as pd

        con = duckdb.connect(database = "f1-race-data-2023.duckdb", read_only = False)
        con.execute(
            """
            ALTER TABLE results_store
            ADD COLUMN IF NOT EXISTS driverExperience INT DEFAULT 0;

            UPDATE results_store
            SET driverExperience = subquery.cumulative_experience
            FROM (
                SELECT 
                    rs.driverId,
                    r.raceId,
                    CASE WHEN ROW_NUMBER() OVER (PARTITION BY rs.driverId ORDER BY r.date DESC) <= 60 THEN 1 ELSE 0 END AS cumulative_experience
                FROM results_store AS rs
                INNER JOIN races AS r ON rs.raceId = r.raceId
            ) AS subquery
            WHERE results_store.driverId = subquery.driverId AND results_store.raceId = subquery.raceId;
            """
        )

        con.execute(
            """
            SELECT
                drivers.driverId,
                drivers.surname,
                drivers.forename,
                races.name,
                races.year,
                races.date,
                results_store.driverExperience,
            FROM results_store
            INNER JOIN drivers ON results_store.driverId = drivers.driverId
            INNER JOIN races ON results_store.raceId = races.raceId
            WHERE surname = 'Leclerc' AND races.year = 2023
            ORDER BY races.date ASC 
            """
        ).df()

        print(
        f"""
        ***************************************************************************

            Feature Engineering - Driver Experience

        ***************************************************************************

            Driver's experience in Formula 1, where a more experienced F1 driver
            typically places better than a rookie.

            Added new feature: 'DriverExperience' to results_store table.
        """
        )

        self.next(self.end)

    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """

if __name__ == "__main__":
    F1PredictionFeaturePipeline()