import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.1

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.002

# Nullspace P gain.
Kn = np.asarray([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0])

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785


# Additional variables for gripper control
GRIPPER_CLOSED_POS = 0.0  # Fully closed
GRIPPER_OPEN_POS = 0.04    # Fully open

def close_gripper(data, actuator_id):
    """Closes the gripper by setting the actuator position."""
    data.ctrl[actuator_id] = GRIPPER_CLOSED_POS

def open_gripper(data, actuator_id):
    """Opens the gripper by setting the actuator position."""
    data.ctrl[actuator_id] = GRIPPER_OPEN_POS

def random_target_positions():
    """Generate random target positions for both arms."""
    x_range = (0.5, 0.8)  # X range
    y_range = (-0.2, 0.2)  # Y range
    z_range = (0.9, 1.2)  # Z range

    target_left = np.array([
        np.random.uniform(*x_range),
        np.random.uniform(*y_range) - 0.15,  # Offset for left arm
        np.random.uniform(*z_range)
    ])

    target_right = np.array([
        np.random.uniform(*x_range),
        np.random.uniform(*y_range) + 0.15,  # Offset for right arm
        np.random.uniform(*z_range)
    ])

    return target_left, target_right

def randomize_cubes(model, data ,mocap_id_left, mocap_id_right):
    cube_positions = [
        np.random.uniform([0.5, -0.2, 0.9], [0.8, 0.2, 1.2]),
        np.random.uniform([0.5, -0.2, 0.9], [0.8, 0.2, 1.2])
    ]
    for i, cube_name in enumerate(["cube1", "cube2"]):
        data.mocap_pos[mocap_id_left] = cube_positions[0]  # np.array([0.7, -0.15, 1.1])
        data.mocap_pos[mocap_id_right] = cube_positions[1] # np.array([0.7, 0.15, 1.1])

def main() -> None:

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path(
        "/Users/rishabhoswal/Documents/Ai2/bimanual-franka-cubepickup_task_submission/resources/bi-franka/scene_with_line_and_2_cubes.xml"
    )
    data = mujoco.MjData(model)

    # Enable gravity compensation.
    model.body_gravcomp[:] = float(gravity_compensation)
    model.opt.timestep = dt

    # Joint and actuator names for both arms.
    left_joint_names = [f"fa-joint{i}" for i in range(1, 8)]
    right_joint_names = [f"fb-joint{i}" for i in range(1, 8)]

    left_actuator_names = [f"fa-actuator{i}" for i in range(1, 8)]
    right_actuator_names = [f"fb-actuator{i}" for i in range(1, 8)]

    # Get joint and actuator IDs correctly.
    dof_ids_left = np.array([model.joint(name).qposadr[0] for name in left_joint_names])
    dof_ids_right = np.array([model.joint(name).qposadr[0] for name in right_joint_names])

    actuator_ids_left = np.array([model.actuator(name).id for name in left_actuator_names])
    actuator_ids_right = np.array([model.actuator(name).id for name in right_actuator_names])

    # Initial joint positions (home keyframe).
    key_id = model.key("home").id
    q0_left = model.key("home").qpos[dof_ids_left]
    q0_right = model.key("home").qpos[dof_ids_right]

    # Mocap targets for both arms.
    mocap_id_left = model.body("target_left").mocapid[0]
    mocap_id_right = model.body("target_right").mocapid[0]



    # Get new random target positions
    target_left, target_right = random_target_positions()

    # Pre-allocate numpy arrays.
    jac_left, jac_right = np.zeros((6, model.nv)), np.zeros((6, model.nv))
    diag = damping * np.eye(6)
    eye = np.eye(model.nv)
    twist_left, twist_right = np.zeros(6), np.zeros(6)
    body_quat_left, body_quat_right = np.zeros(4), np.zeros(4)
    body_quat_conj_left = np.zeros(4)
    body_quat_conj_right = np.zeros(4)
    error_quat_left, error_quat_right = np.zeros(4), np.zeros(4)
    dq = np.zeros(model.nv)

    # Get joint and actuator names
    left_gripper_actuator = model.actuator("fa-actuator8").id
    right_gripper_actuator = model.actuator("fb-actuator8").id

    gripper_closed = False  # Track if grippers are closed
    grasp_threshold = 0.02

    # Set mocap positions.
    cube_positions = [
        np.random.uniform([0.3, -0.5, 0.98], [0.8, 0.0, 0.98]),
        np.random.uniform([0.3, 0.0, 0.98], [0.8, 0.5, 0.98])
    ]
    
    print(cube_positions)


    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # Reset simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)


        for i, cube_name in enumerate(["cube_left", "cube_right"]):
            data.body(cube_name).xpos[:] = cube_positions[i]

        # Set mocap positions dynamically (0.2 above the cube)
        data.mocap_pos[mocap_id_left] = cube_positions[0] + np.array([0, 0, 0.10])
        data.mocap_pos[mocap_id_right] = cube_positions[1] + np.array([0, 0, 0.10])

        # New cube positions
        # cube_left_pos = [0.6, -0.2, 0.98]  # Update this dynamically
        # cube_right_pos = [0.7, 0.2, 0.98]  # Update this dynamically

        # Update the keyframe qpos for the cubes
        # print(data.qpos[31])   # 0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04 0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04 0.7 -0.15 0.98 1 0 0 0 0.7 0.15 0.98 1 0 0 0
        data.qpos[18:21] = cube_positions[0]  # Adjust indices as needed
        data.qpos[25:28] = cube_positions[1]  # Adjust indices as needed

        mujoco.mj_forward(model, data)

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        body_id_left = model.body("fa-hand").id
        body_id_right = model.body("fb-hand").id

        # Set up the video recording using OpenCV.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec
        video_out = cv2.VideoWriter('simulation_video.mp4', fourcc, 30, (640, 480))  # Video file output

        time.sleep(3)

        while viewer.is_running():
            step_start = time.time()

            # --- Left Hand ---
            dx_left = data.mocap_pos[mocap_id_left] - data.body("fa-hand").xpos
            twist_left[:3] = Kpos * dx_left / integration_dt
            body_quat_left[:] = data.body("fa-hand").xquat
            mujoco.mju_negQuat(body_quat_conj_left, body_quat_left)
            mujoco.mju_mulQuat(error_quat_left, data.mocap_quat[mocap_id_left], body_quat_conj_left)
            mujoco.mju_quat2Vel(twist_left[3:], error_quat_left, 1.0)
            twist_left[3:] *= Kori / integration_dt

            mujoco.mj_jacBody(model, data, jac_left[:3], jac_left[3:], body_id_left)
            dq[dof_ids_left] = jac_left[:, dof_ids_left].T @ np.linalg.solve(
                jac_left[:, dof_ids_left] @ jac_left[:, dof_ids_left].T + diag, twist_left
            )

            dq[dof_ids_left] += (np.eye(7) - np.linalg.pinv(jac_left[:, dof_ids_left]) @ jac_left[:, dof_ids_left]) @ (
                Kn * (q0_left - data.qpos[dof_ids_left])
            )

            # --- Right Hand ---
            dx_right = data.mocap_pos[mocap_id_right] - data.body("fb-hand").xpos
            twist_right[:3] = Kpos * dx_right / integration_dt
            body_quat_right[:] = data.body("fb-hand").xquat
            mujoco.mju_negQuat(body_quat_conj_right, body_quat_right)
            mujoco.mju_mulQuat(error_quat_right, data.mocap_quat[mocap_id_right], body_quat_conj_right)
            mujoco.mju_quat2Vel(twist_right[3:], error_quat_right, 1.0)
            twist_right[3:] *= Kori / integration_dt

            mujoco.mj_jacBody(model, data, jac_right[:3], jac_right[3:], body_id_right)
            dq[dof_ids_right] = jac_right[:, dof_ids_right].T @ np.linalg.solve(
                jac_right[:, dof_ids_right] @ jac_right[:, dof_ids_right].T + diag, twist_right
            )

            dq[dof_ids_right] += (np.eye(7) - np.linalg.pinv(jac_right[:, dof_ids_right]) @ jac_right[:, dof_ids_right]) @ (
                Kn * (q0_right - data.qpos[dof_ids_right])
            )

            # Clamp max joint velocity
            dq_abs_max = np.abs(dq).max()
            if dq_abs_max > max_angvel:
                dq *= max_angvel / dq_abs_max

            # Integrate joint velocities to obtain joint positions.
            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)

            # Set control signal.
            data.ctrl[actuator_ids_left] = q[dof_ids_left]
            data.ctrl[actuator_ids_right] = q[dof_ids_right]  # <-- FIXED

            # Check if hands reached the grasping position and close grippers
            if not gripper_closed and np.linalg.norm(dx_left) < grasp_threshold and np.linalg.norm(dx_right) < grasp_threshold:
                close_gripper(data, left_gripper_actuator)
                close_gripper(data, right_gripper_actuator)
                gripper_closed = True  # Prevent re-closing

            # Step the simulation.
            mujoco.mj_step(model, data)
            viewer.sync()

            # Capture frame and write it to video
            # Capture and write the frame to the video.
            # img = np.array(viewer.render(width=640, height=480))  # Adjust image size as needed
            # video_writer.write(img)

            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


        # # Close grippers to grasp cubes
        # close_gripper(data, left_gripper_actuator)
        # close_gripper(data, right_gripper_actuator)

        # time.sleep(1)  # Wait for grasp

        # # Move to new target positions
        # data.mocap_pos[mocap_id_left] = target_left
        # data.mocap_pos[mocap_id_right] = target_right
        # mujoco.mj_forward(model, data)

if __name__ == "__main__":
    main()