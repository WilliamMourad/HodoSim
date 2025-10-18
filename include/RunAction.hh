#pragma once

#include "G4UserRunAction.hh"
#include "G4AnalysisManager.hh"
#include "G4Timer.hh"

struct RunActionParameters {
	G4bool enableCuts;
};

class RunAction : public G4UserRunAction 
{
public:
	RunAction(RunActionParameters runActionParameters);
	~RunAction();

	void BeginOfRunAction(const G4Run* run) override;
	void EndOfRunAction(const G4Run* run) override;

private:
	RunActionParameters _runActionParameters;

	G4AnalysisManager* analysisManager;
	G4Timer* timer;
};