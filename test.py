import os
import mujoco

def convert_urdf_to_mjcf(urdf_path, output_mjcf_path):

   
        
    print(f"Loading and compiling URDF: {urdf_path}")
    
    # 1. Load and compile the URDF file into an MjModel object
    # Note: MuJoCo uses internal path-solving, so make sure mesh paths in your 
    # URDF are either absolute or relative to the URDF's directory location.
    model = mujoco.MjModel.from_xml_path("/home/ishan-agrawal/robotics/mjsim2real/env/so101_latest.urdf")
    
    print(f"Saving compiled model to MJCF: {output_mjcf_path}")
    
    # 2. Serialize the internal representation back out into an MJCF XML file
    mujoco.mj_saveLastXML(output_mjcf_path, model)
    
    print("Conversion complete!")

if __name__ == "__main__":
    # Update these paths to match your local file structure
    input_urdf = "my_robot.urdf"
    output_mjcf = "so101_latest.xml"
    
    convert_urdf_to_mjcf(input_urdf, output_mjcf)