import run_preprocessing as main_pipeline
dataset = 'CHICAGO' # 'NYC'
for traj_len in [25, 20, 150, 10]:
    for radius in [1000, 750, 500, 250]:
        for time_window in [4, 3, 2, 1]:
            main_pipeline.main(dataset=dataset, traj_len=traj_len, crime_radius=radius, crime_time_weeks=time_window)