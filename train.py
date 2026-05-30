

from roboflow import Roboflow
rf = Roboflow(api_key="ED4wbI13fv76CxaIJde4")
project = rf.workspace("ai-traffic-ambulance-detection").project("ambulance-detection-u4ao4-76laq")
version = project.version(1)
dataset = version.download("yolov8")
                