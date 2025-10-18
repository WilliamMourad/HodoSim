#pragma once

#include "G4UserSteppingAction.hh"
#include "G4Track.hh"
#include "G4Step.hh"

struct SteppingActionParameters {
	G4String scintLVName;
};

class SteppingAction : public G4UserSteppingAction
{
public:
	SteppingAction(SteppingActionParameters steppingActionParameters);
	~SteppingAction();
	void UserSteppingAction(const G4Step* step) override;

private:
	
	void ProcessOPReflections(const G4Track* track, const G4Step* step);
	void ProcessMuPosition(const G4Track* track, const G4Step* step);

	SteppingActionParameters _steppingActionParameters;
};