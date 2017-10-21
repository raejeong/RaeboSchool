# RaeboSchool

Implementing cool Deep Reinforcement Learning Algorithms to solve Robotic Learning problems from OpenAI's RoboSchool! Docker is used to setup all the libraries for easy development!

# Installed Software

- Tensorflow
- PyTorch
- OpenAI Gym
- Roboschool
- ROS

# Install

Install docker

Install nvidia-docker

Clone this repo

# Run

- X server, lazy way
```xhost +local:root ```

- build docker image
```sudo docker build . -t raeboschool ```

- run the docker container
```sudo docker stop raeboschool_c; sudo docker rm raeboschool_c; sudo nvidia-docker run -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v $(pwd)/workspace:/home/user/workspace --privileged --net=host --name raeboschool_c raeboschool```

- to run additional terminal for the container

```sudo docker exec -it raeboschool_c bash```

- stop and remove docker container

```sudo docker stop raeboschool_c; sudo docker rm raeboschool_c```
