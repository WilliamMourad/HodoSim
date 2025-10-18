#include "ActionInitialization.hh"



ActionInitialization::ActionInitialization(
	PrimaryGeneratorActionParameters primaryGeneratorActionParameters,
	RunActionParameters runActionParameters,
	EventActionParameters eventActionParameters,
	TrackingActionParameters trackingActionParameters,
	SteppingActionParameters steppingActionParameters
) : G4VUserActionInitialization() 
{
	_primaryGeneratorActionParameters = primaryGeneratorActionParameters;
	_runActionParameters = runActionParameters;
	_eventActionParameters = eventActionParameters;
	_trackingActionParameters = trackingActionParameters;
	_steppingActionParameters = steppingActionParameters;
}

ActionInitialization::~ActionInitialization() {}

// This method is called only in the master thread in multi-threaded mode,
// omitting this brings issues with data collection.
void ActionInitialization::BuildForMaster() const 
{
	SetUserAction(new RunAction(_runActionParameters));
}

void ActionInitialization::Build() const 
{
	
	// instantiate actions here

	auto* eventAction = new EventAction(_eventActionParameters);

    SetUserAction(new PrimaryGeneratorAction(_primaryGeneratorActionParameters));
    SetUserAction(new RunAction(_runActionParameters));
	SetUserAction(eventAction);
	SetUserAction(new TrackingAction(_trackingActionParameters, eventAction));
	SetUserAction(new SteppingAction(_steppingActionParameters));
}
