@startuml
title BLE State Machine

[*] --> Initialize
    
Initialize--> Logging :Success
Initialize :Initialize BLE interface

Logging --> Stopped :Application shutting down
Logging--> Send data :TimeOut
    
Logging :Start polling loop
Logging :Start Timer
Logging :Log sensor data
	
Send data --> Logging :Finished
Send data :If log not empty, put data into que and create data event
    
Stopped:Cleanup BLE interface;


@enduml
