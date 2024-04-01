import pickle
from helper_functions import record_trial, analyze_trial

timeVector = record_trial(record_length=30, size=[1920, 1080], frame_rate=30)

# fps = int(len(timeVector)/((timeVector[-1] - timeVector[0])))

markers_coords = analyze_trial(baseline=25.4, record_video='Video_Files/03_30_2024/drop_4mph_test2.avi', display_video=True, minArea=400, 
                               markerColors=['neon_green', 'neon_pink', 'yellow'], cameras_angle=0, size=[1920, 1080], frame_rate=30)

with open('Mocap_3D_Trials/03_30_2024/drop_4mph_test2', 'wb') as handle:
    pickle.dump({'markerVector' : markers_coords, 'timeVector' : timeVector}, handle, protocol=pickle.HIGHEST_PROTOCOL)
