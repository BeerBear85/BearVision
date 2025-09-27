@startuml
title BearVision Runtime State Machine

[*] --> Initialize

Initialize --> LookingForWakeboarder : Initialization complete
Initialize --> Error : Initialization error
Initialize:Connect to GoPro
Initialize:Start GoPro preview
Initialize:Start Bluetooth logging thread
Initialize:Start GUI thread
Initialize:Start Cloud upload thread
Initialize:Start Post-processing thread

LookingForWakeboarder --> Recording : Wakeboarder detected
LookingForWakeboarder --> Error : Detection error
LookingForWakeboarder:Enable Hindsight on GoPro
LookingForWakeboarder:Start YOLO detection loop

Recording --> LookingForWakeboarder : After clip capture
Recording --> Error : Recording error
Recording:Trigger recording;
Recording:Send clip to post-processing thread;

Stopping:aaa
Stopping:Close threads

Error --> Initialize : Restart system
Error:Display/log error;


@enduml
