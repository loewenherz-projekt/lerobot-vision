import os
import sys
import types

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

# rclpy stub
if "rclpy" not in sys.modules:
    rclpy = types.ModuleType("rclpy")
    node_module = types.ModuleType("rclpy.node")

    class Node:
        def __init__(self, name: str):
            self.name = name

        def declare_parameter(self, name: str, value=None):
            return value

        def get_parameter(self, name: str):
            class Param:
                def __init__(self, value):
                    self._value = value

                def get_parameter_value(self):
                    class V:
                        def __init__(self, v):
                            self.string_value = v

                    return V(self._value)

            return Param(None)

        def create_publisher(self, *args, **kwargs):
            class Pub:
                def publish(self, msg):
                    pass

            return Pub()

        def create_subscription(self, *args, **kwargs):
            return lambda msg: None

        def create_timer(self, *args, **kwargs):
            return None

    node_module.Node = Node
    rclpy.node = node_module
    rclpy.init = lambda args=None: None
    rclpy.shutdown = lambda: None
    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = node_module

# cv_bridge stub
if "cv_bridge" not in sys.modules:
    cv_bridge = types.ModuleType("cv_bridge")

    class CvBridge:
        def cv2_to_imgmsg(self, img, encoding="bgr8"):
            return object()

    cv_bridge.CvBridge = CvBridge
    sys.modules["cv_bridge"] = cv_bridge

# message stubs
for mod_name in [
    "sensor_msgs.msg",
    "vision_msgs.msg",
    "std_msgs.msg",
    "trajectory_msgs.msg",
]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)

        class Msg:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        m.Image = Msg
        m.PointCloud2 = Msg
        m.Detection3DArray = Msg
        m.String = Msg
        m.JointTrajectory = Msg
        m.JointTrajectoryPoint = Msg
        sys.modules[mod_name] = m
