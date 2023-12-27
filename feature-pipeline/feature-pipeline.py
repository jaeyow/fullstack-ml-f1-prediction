# pylint: disable=C0415,C0103,W0201,W0702,W0718
import os
from metaflow import FlowSpec, step
# from comet_ml.integration.metaflow import comet_flow

# try:
#     from dotenv import load_dotenv

#     load_dotenv(verbose=True, dotenv_path=".env")
# except Exception:
#     print("No dotenv package")


# @comet_flow(project_name="<please-replace-me>")
class F1PredictionFeaturePipeline(FlowSpec):

    @step
    def start(self):
        """
        Initialization, place everything init related here, check that everything is
        in order like environment variables, connection strings, etc, and if there are
        any issues, quit now.
        """
        assert os.environ["FS_API_KEY"]
        assert os.environ["FS_PROJECT_NAME"]

        self.FS_API_KEY = os.environ["FS_API_KEY"]
        self.FS_PROJECT_NAME = os.environ["FS_PROJECT_NAME"]

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
                driver_standings.points,
                races.name,
                races.year
            FROM drivers
            INNER JOIN driver_standings
                ON drivers.driverId = driver_standings.driverId
            INNER JOIN races
                ON driver_standings.raceId = races.raceId
            WHERE surname = 'Ricciardo' AND driver_standings.position <= 5
            ORDER BY driver_standings.points DESC;
            """
        ).df()
        print(drv_df.head(5))

        results_store_df = con.execute(
            """
            SELECT * FROM results_store;
            """
        ).df()
        print(results_store_df.head(5))
        
        self.next(self.create_drivers_experience_table)

    @step
    def create_drivers_experience_table(self):
        """
        create_drivers_experience_table
        """
        import duckdb
        import numpy as np
        import pandas as pd

        print("create_drivers_experience_table...")
        con = duckdb.connect(database = "f1-race-data-2023.duckdb", read_only = False)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS driver_experience AS (
                WITH RankedResults AS (
                    SELECT
                        driv.driverId,
                        driv.surname,
                        COUNT(*) OVER (PARTITION BY driv.driverId ORDER BY rac.date) AS race_count
                    FROM
                        main.results AS res
                    INNER JOIN
                        main.races AS rac ON res.raceId = rac.raceId 
                    INNER JOIN
                        main.drivers AS driv ON res.driverId = driv.driverId
                    WHERE
                        rac.date >= CURRENT_DATE - INTERVAL '5' YEAR    
                )
                SELECT
                    driverId,
                    surname,
                    MAX(race_count) AS total_races
                FROM
                    RankedResults
                GROUP BY
                    driverId, surname
                ORDER BY total_races DESC
            )
            """
        )
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
            SET driverExperience = subquery.total_races
            FROM (
                SELECT DISTINCT
                    driv.driverId as drivId,
                    driv_exp.total_races as total_races
                FROM main.results AS res
                INNER JOIN
                    main.drivers AS driv ON res.driverId = driv.driverId
                INNER JOIN 
                    main.driver_experience as driv_exp ON res.driverId = driv_exp.driverId
                ) AS subquery
            WHERE main.results_store.driverId = subquery.drivId;
            """
        )

        results_store_df = con.execute("SELECT * FROM results_store;").df()
        self.results_store = results_store_df

        print(
        f"""
        ***************************************************************************

            Feature Engineering - Driver Experience

        ***************************************************************************

            Driver's experience in Formula 1, where a more experienced F1 driver
            typically places better than a rookie.

            Added new feature: 'DriverExperience' to results_store table. {self.results_store.shape}
            
        """
        )

        self.next(self.create_constructor_experience_table)

    @step
    def create_constructor_experience_table(self):
        """
        create_constructor_experience_table
        """
        import duckdb
        import numpy as np
        import pandas as pd

        print("create_constructor_experience_table...")
        con = duckdb.connect(database = "f1-race-data-2023.duckdb", read_only = False)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS constructor_experience AS (
                WITH RankedResults AS (
                    SELECT
                        cons.constructorId AS constructorId,
                        cons.name AS name,
                        COUNT(*) OVER (PARTITION BY cons.constructorId ORDER BY rac.date) AS race_count
                    FROM
                        main.results AS res
                    INNER JOIN
                        main.races AS rac ON res.raceId = rac.raceId
                    INNER JOIN
                        main.constructors AS cons ON res.constructorId = cons.constructorId
                    WHERE
                        rac.date >= CURRENT_DATE - INTERVAL '5' YEAR
                )
                SELECT
                    constructorId,
                    name,
                    MAX(race_count) AS total_races
                FROM
                    RankedResults
                GROUP BY
                    constructorId, name
                ORDER BY total_races DESC
            )
            """
        )
        self.next(self.create_constructor_experience_feature)

    @step
    def create_constructor_experience_feature(self):
        """
        Feature Engineering: Constructor's experience in Formula 1, where a more experienced F1
        constructor typically places better than a rookie. 
        """
        import duckdb
        import numpy as np
        import pandas as pd

        con = duckdb.connect(database = "f1-race-data-2023.duckdb", read_only = False)
        con.execute(
            """
            ALTER TABLE results_store
            ADD COLUMN IF NOT EXISTS constructorExperience INT DEFAULT 0;

            UPDATE results_store
            SET constructorExperience = subquery.total_races
            FROM (
                SELECT DISTINCT
                    cons.constructorId as constructorId,
                    cons_exp.total_races as total_races
                FROM
                    main.results AS res
                INNER JOIN
                    main.races AS rac ON res.raceId = rac.raceId
                INNER JOIN 
                    main.constructor_experience as cons_exp ON res.constructorId = cons_exp.constructorId    
                INNER JOIN
                    main.constructors AS cons ON res.constructorId = cons.constructorId
            ) AS subquery
            WHERE results_store.constructorId = subquery.constructorId;
            """
        )      

        results_store_df = con.execute("SELECT * FROM results_store;").df()
        self.results_store = results_store_df

        print(
        f"""
        ***************************************************************************

            Feature Engineering - Constructor Experience

        ***************************************************************************

            Constructor's experience in Formula 1, where a more experienced F1
            constructor typically places better than a rookie.

            Added new feature: 'ConstructorExperience', new dataframe shape: {self.results_store.shape}
        """
        )

        self.next(self.finalise_feature_list)  

    @step
    def finalise_feature_list(self):
        """
        Whittle down the feature list to the most important features.
        """
        self.results_store.drop(['resultId','number','positionText','positionOrder','time','milliseconds'], axis=1, inplace=True)

        self.next(self.create_feature_group)

    @step
    def create_feature_group(self):
        """
        create_feature_group
        """
        import hopsworks
        import pandas as pd
        # from great_expectations.core import ExpectationSuite
        # from hsfs.feature_group import FeatureGroup

        # Connect to feature store.
        project = hopsworks.login(
            api_key_value=self.FS_API_KEY, project=self.FS_PROJECT_NAME
        )
        feature_store = project.get_feature_store()

        # Create feature group.
        f1_prediction_fg = feature_store.get_or_create_feature_group(
            name="f1_prediction_fg_1",
            primary_key=["driverId"],
            online_enabled=True,
            version=3,
            description="Blah blah blah",
            time_travel_format=None,
        )
        # Upload data.
        f1_prediction_fg.insert(
            features=self.results_store,
            overwrite=False,
            write_options={
                "wait_for_job": True,
            },
        )

        # Add feature descriptions.
        # resultId,raceId,driverId,constructorId,number,grid,position,positionText,positionOrder,points,
        # laps,time,milliseconds,fastestLap,rank,fastestLapTime,fastestLapSpeed,statusId
        # feature_descriptions = [
        #     {
        #         "name": "datetime_utc",
        #         "description": """
        #                         Datetime interval in UTC when the data was observed.
        #                         """,
        #         "validation_rules": "Always full hours, i.e. minutes are 00",
        #     },
        #     {
        #         "name": "area",
        #         "description": """
        #                         Denmark is divided in two price areas, divided by the Great Belt: DK1 and DK2.
        #                         If price area is “DK”, the data covers all Denmark.
        #                         """,
        #         "validation_rules": "0 (DK), 1 (DK1) or 2 (Dk2) (int)",
        #     },
        #     {
        #         "name": "consumer_type",
        #         "description": """
        #                         The consumer type is the Industry Code DE35 which is owned by Danish Energy. 
        #                         The code is used by Danish energy companies.
        #                         """,
        #         "validation_rules": ">0 (int)",
        #     },
        #     {
        #         "name": "energy_consumption",
        #         "description": "Total electricity consumption in kWh.",
        #         "validation_rules": ">=0 (float)",
        #     },
        # ]
        # for description in feature_descriptions:
        #     energy_feature_group.update_feature_description(
        #         description["name"], description["description"]
        #     )

        # Update statistics.
        f1_prediction_fg.statistics_config = {
            "enabled": True,
            "histograms": True,
            "correlations": True,
        }
        f1_prediction_fg.update_statistics_config()
        f1_prediction_fg.compute_statistics()

        self.next(self.end)


    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """

if __name__ == "__main__":
    F1PredictionFeaturePipeline()