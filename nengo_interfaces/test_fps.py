import numpy as np
dt = 0.005
camera_params = {
    'camera_name': '0',
    'save_name': 'data/images/drone_fpv',
    'fps': 33.3333
}

if camera_params:
    max_fps = np.round(1/dt, 4)
    assert camera_params['fps'] <= max_fps, (
        'With a dt of %.3f sec your maximum camera fps is %.2f' % (
            dt, 1/dt)
        )
    fps_remainder = (1/camera_params['fps']) - int(1/camera_params['fps'])
    fps_count = 0

    frame_rate_locked = False
    available_frame_rates = []
    fps_multiple = 1
    while max_fps >= camera_params['fps']:
        available_frame_rates.append(max_fps)

        if max_fps == camera_params['fps']:
            frame_rate_locked = True
            break
        else:
            fps_multiple += 1
            max_fps = np.round(1/(dt*fps_multiple), 4)
    if not frame_rate_locked:
        raise ValueError ('Please select an fps with a time/image that is a multiple of your timestep:', available_frame_rates)

t = 0
# print('time per image: ', np.round((1/camera_params['fps']), 4))
# import time
# time.sleep(10)
while t < 10:
    t = np.round(t+dt, 4)
    if camera_params:
        # subtract dt because we start at dt, not zero, but we want an image on the first step, before the drone starts moving
        # print('frame time: ', 1/camera_params['fps'])
        # print('mod: ', (t-dt) % camera_params['fps'])
        # print(np.round(1/camera_params['fps'], 4))
        # print(1/camera_params['fps'])
        # print('t-dt: ', t-dt)

        blue = '\033[94m'
        endc = '\033[0m'
        green = '\033[92m'
        red = '\033[91m'
        # print('modulo: ', (np.round(t-dt, 4)) % np.round(1/camera_params['fps'], 4))
        if (int(1000*np.round(t-dt, 4))) % int(1000*np.round(1/camera_params['fps'], 4)) < 1e-5:
        # if (t-dt) % (1/camera_params['fps']) <= fps_remainder:
            fps_count += 1
            # print('%simg captured%s'%(green, endc))
            # print('t: ', t-dt)

        if t%1 == 0:
            print('%s1 sec passed, recorded %i images%s' % (red, fps_count, endc))
            fps_count = 0
        # else:
        #     print(t)


