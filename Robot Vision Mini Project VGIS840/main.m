%% RobotVision Camera sutff

clc;
close all;

% Includes helper functions and FIXED Robolink class
addpath('functions');
addpath('source');

% Link MATLAB with RoboDK
RDK = Robolink;

% Set the simulation speed. Thisis a ratio, for example, simulation speed
% of 5 (default) means that 1 second of simulated time corresponds to 1
% second of real time.
RDK.setSimulationSpeed(5);

% Define the robot
arm = RDK.Item('KUKA KR 6 R700 sixx');
arm.setSpeed(100); % Set the TCP linear speed in mm/s

reset = RDK.Item('Reset');
reset.RunProgram();

% Get world frame
worldCoord = RDK.Item('World');
arm.setPoseFrame(worldCoord);

% Get build frame and its position
builderCoord = RDK.Item('LegoBuilder');

% Get reference to the tool TCP
tool = RDK.Item('GripperTCP');

% Get reference to the gripper
gripper = RDK.Item('RobotiQ 2F85');

HomePos = RDK.Item('Home');
robotHome = HomePos.Joints();
% Move robot to the home position (defined in the simulation)
arm.MoveJ(robotHome);


% Initiate robot helper functions with references to the items
robot = Robot;
robot.roboarm = arm; 
robot.robotool = tool;
robot.robogripper = gripper;

% Get the center point of the plate
BuildingPos = builderCoord.Pose() + robot.addTrans(130, 130, 30);


% initate the camera
RDK.Cam2D_Close(0);

camera = RDK.Item('Camera');
camera_id = RDK.Cam2D_Add(camera);

% Take a picture and close again
RDK.Cam2D_Snapshot('C:\Users\Elina\Desktop\Robot Vision Mini Project VGIS840\images\original.jpg', camera_id);
RDK.Cam2D_Close(0);


% Import the python script for image processing
import py.Robot_vision.*

%TODO make it run when you misspell as well
character = input('Please enter the name of the character that you want to build(homer, marge, bart, maggie or lisa) \n','s');

if isempty(character)
    disp('miss spelled run the program again');
    
else

    disp('Building your Character');
    
    % Call the Python script with the wanted character name
    results = py.Robot_vision.run(character); 

    % Convert the robotvision to a matrix
    table = cell(results);

    coordinates = zeros(length(table), 2);

    for i = 1:length(table)
        array = cell(table{i});
        coordinates(i, :) = cellfun(@double, array);
    end
      
    disp(coordinates);
    

    
    for i = 1:length(coordinates)
        % Get the target 
        pos = coordinates(i,:); 
        
        % Move to target positon
        robot.setXYZ( pos(1,1) , pos(1,2), 35 );
        
        % Pickup the brick
        robot.attach();
        
        % Move up a little 
        robot.moveZ(35);
        
        % Move to the building zone (a little above)
        pos = BuildingPos + robot.addTrans(0,0, (i-1)*20 + 30);
        robot.setTrans(pos);
        
        % Lower to correct height
        robot.moveZ(-30);
        
        % Detach the brick
        robot.detach();
        
        % Raise the tool again
        robot.moveZ(30);
 
    end
    
    arm.MoveJ(robotHome);
        
end
