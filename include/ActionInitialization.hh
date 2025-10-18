#pragma once

#include "G4VUserActionInitialization.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "TrackingAction.hh"
#include "SteppingAction.hh"


// I defined the following structs in the relative header file to make it easier to pass parameters 
// directly from the action initialization that happens in main.cc
// struct PrimaryGeneratorActionParameter
// struct RunActionParameters
// struct EventActionParameters
// struct TrackingActionParameters
// struct SteppingActionParameters


class ActionInitialization : public G4VUserActionInitialization 
{
public:

    ActionInitialization(
        PrimaryGeneratorActionParameters primaryGeneratorActionParameters,
        RunActionParameters runActionParameters,
        EventActionParameters eventActionParameters,
        TrackingActionParameters trackingActionParameters,
        SteppingActionParameters steppingActionParameters
    );
    
    ~ActionInitialization();

	void BuildForMaster() const override;
    void Build() const override;

private:

    PrimaryGeneratorActionParameters _primaryGeneratorActionParameters;
    RunActionParameters _runActionParameters;
    EventActionParameters _eventActionParameters;
	TrackingActionParameters _trackingActionParameters;
	SteppingActionParameters _steppingActionParameters;

};
