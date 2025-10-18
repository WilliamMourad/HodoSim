#include "PrimaryGeneratorAction.hh"

#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4GeneralParticleSource.hh"


PrimaryGeneratorAction::PrimaryGeneratorAction(PrimaryGeneratorActionParameters primaryGeneratorActionParameters) {
    
	_primaryGeneratorActionParameters = primaryGeneratorActionParameters;

	// I like modularity, it keeps things tidy and easier to debug.
    BuildParticleGun();
    BuildGPS();
}

void PrimaryGeneratorAction::BuildParticleGun()
{
    // Im not writing lots of comments here because i think the code is pretty self explanatory.
	
    G4String particleName = _primaryGeneratorActionParameters.particleName;
    G4ParticleDefinition* particle = G4ParticleTable::GetParticleTable()->FindParticle(particleName);

    auto particleGunSettings = _primaryGeneratorActionParameters.particleGunSettings;
	
    if (!particleGunSettings.isActive) return;

    G4int particleN = particleGunSettings.particleN;
    G4double particleEnergy = particleGunSettings.particleEnergy;
    G4ThreeVector gunPosition = particleGunSettings.gunPosition;
    G4ThreeVector gunDirection = particleGunSettings.gunDirection;
    
    particleGun = new G4ParticleGun(particleN);
    particleGun->SetParticleDefinition(particle);
    particleGun->SetParticlePosition(gunPosition);
    particleGun->SetParticleMomentumDirection(gunDirection);
    particleGun->SetParticleEnergy(particleEnergy);
}

// From my experience when possible it is better to set gps parameters using a macro file, (since the documentation on how to do it in code is very poor)
// but since this simulation requires a simple beam, there is no harm in doing it here.
// Doing it in code also makes it easier to tweak the parameters from the main.cc file.
void PrimaryGeneratorAction::BuildGPS()
{
    // Im not writing lots of comments here because i think the code is pretty self explanatory.
    // Anyway, to learn more on GPS check https://www.fe.infn.it/u/paterno/Geant4_tutorial/slides_further/GPS/GPS_manual.pdf
    
    G4String particleName = _primaryGeneratorActionParameters.particleName;
    G4ParticleDefinition* particle = G4ParticleTable::GetParticleTable()->FindParticle(particleName);

    auto gpsSettings = _primaryGeneratorActionParameters.gpsSettings;

    if (!gpsSettings.isActive) return;

    G4int particleN = 1;
    G4double particleMeanEnergy = gpsSettings.particleMeanEnergy;
	G4double particleEnergySigma = gpsSettings.particleEnergySigma;
	G4ThreeVector gpsPosition = gpsSettings.gpsPosition;
	G4double gpsRadius = gpsSettings.gpsRadius;
	G4ThreeVector gpsRot1 = gpsSettings.gpsRot1;
	G4ThreeVector gpsRot2 = gpsSettings.gpsRot2;
	G4double beamApertureX = gpsSettings.beamApertureX;
	G4double beamApertureY = gpsSettings.beamApertureY;

    gps = new G4GeneralParticleSource();

    gps->SetParticleDefinition(particle);
	gps->GetCurrentSource()->SetNumberOfParticles(1);
	
    // Set a gaussian energy distribution
    gps->GetCurrentSource()->GetEneDist()->SetEnergyDisType("Gauss");
	gps->GetCurrentSource()->GetEneDist()->SetMonoEnergy(particleMeanEnergy);
	gps->GetCurrentSource()->GetEneDist()->SetBeamSigmaInE(particleEnergySigma);
    
    // set position of gps
	gps->GetCurrentSource()->GetPosDist()->SetPosDisType("Plane");
    gps->GetCurrentSource()->GetPosDist()->SetPosDisShape("Circle");
	gps->GetCurrentSource()->GetPosDist()->SetCentreCoords(gpsPosition);
	gps->GetCurrentSource()->GetPosDist()->SetRadius(gpsRadius);

    // set rotation of gps
    // It took me some time to figure out how to do this!
    gps->GetCurrentSource()->GetAngDist()->DefineAngRefAxes("angref1", gpsRot1);
    gps->GetCurrentSource()->GetAngDist()->DefineAngRefAxes("angref2", gpsRot2);
    gps->GetCurrentSource()->GetAngDist()->SetUseUserAngAxis(true);
    gps->GetCurrentSource()->GetAngDist()->SetParticleMomentumDirection({0., 0., 1.});
    
    // set angular distribution of gps
	gps->GetCurrentSource()->GetAngDist()->SetAngDistType("beam2d");
	gps->GetCurrentSource()->GetAngDist()->SetBeamSigmaInAngX(beamApertureX); // radians
	gps->GetCurrentSource()->GetAngDist()->SetBeamSigmaInAngY(beamApertureY); // radians
}

PrimaryGeneratorAction::~PrimaryGeneratorAction() {
    delete particleGun;
    delete gps;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent) {
    
    auto particleGunSettings = _primaryGeneratorActionParameters.particleGunSettings;
    auto gpsSettings = _primaryGeneratorActionParameters.gpsSettings;
    
    if (particleGunSettings.isActive) particleGun->GeneratePrimaryVertex(anEvent);
    if (gpsSettings.isActive) gps->GeneratePrimaryVertex(anEvent);
}

