#include <TMap.h>
#include <TTree.h>
#include "TVirtualFFT.h"
#include <TGraph.h>
#include <TCanvas.h>
#include <TCutG.h>
#include <TH2D.h>
#include <THStack.h>
#include <TPaletteAxis.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TH1F.h>
#include <TAxis.h>
#include <TF1.h>
#include <TFile.h>
#include <TImage.h>
#include <TColor.h>
#include "TSpectrum.h"
#include <TVector3.h>
#include <vector>
#include "TPolyMarker.h"

void PlotRoot(){
  TTree * tree = new TTree("tree","tree");
  tree->ReadFile("/home/kswiss/Workspace/worktorch/ELVESNet/output_171004/outputForROOTB.txt","name/C:lattrue/F:latpred/F:lontrue/F:lonpred/F:intrainFOV/I");
  TH2F * histlat = new TH2F("histlat","Latitude MSE Heat Map in Truth Coord. (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlon = new TH2F("histlon","Longitude MSE Heat Map in Truth Coord. (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlatcount = new TH2F("histlatcount","Counts (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histloncount = new TH2F("histloncount","Counts (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlatcountzoom = new TH2F("histlatcountzoom","Counts (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histloncountzoom = new TH2F("histloncountzoom","Counts (All Events);True Longitude;True Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlatp = new TH2F("histlatp","Latitude MSE Heat Map in Predited Coord. (All Events);Predicted Longitude;Predicted Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlonp = new TH2F("histlonp","Longitude MSE Heat Map in Predicted Coord. (All Events);Predicted Longitude;Predicted Latitude",16,-68,-60,16,-38,-30);
  TH2F * histlatzoom = new TH2F("histlatzoom","Latitude MSE Heat Map in Truth Coord. (w/in train range);True Longitude;True Latitude",16,-68,-63,16,-37,-32);
  TH2F * histlonzoom = new TH2F("histlonzoom","Longitude MSE Heat Map in Truth Coord. (w/in train range);True Longitude;True Latitude",16,-68,-63,16,-37,-32);
  TH2F * histlatpzoom = new TH2F("histlatpzoom","Latitude MSE Heat Map in Predited Coord. (w/in train range);Predicted Longitude;Predicted Latitude",16,-68,-63,16,-37,-32);
  TH2F * histlonpzoom = new TH2F("histlonpzoom","Longitude MSE Heat Map in Predicted Coord. (w/in train range);Predicted Longitude;Predicted Latitude",16,-68,-63,16,-37,-32);
  TH2F * histlatconf = new TH2F("histlatconf","Confusion Matrix;Predicted Latitude;True Latitude",40,-38,-30,40,-38,-30);
  TH2F * histlonconf = new TH2F("histlonconf","Confusion Matrix;Predicted Longitude;True Longitude",40,-68,-60,40,-68,-60);
  TGraph *glatconf = new TGraph(tree->GetEntries());
  TGraph *glonconf = new TGraph(tree->GetEntries());
  TGraph *glatconfOUT = new TGraph(tree->GetEntries());
  TGraph *glonconfOUT = new TGraph(tree->GetEntries());
  float lattrue,lontrue,latpred,lonpred;
  float laterro,lonerro;
  int intrainFOV;
  
  tree->SetBranchAddress("lattrue",&lattrue);
  tree->SetBranchAddress("lontrue",&lontrue);
  tree->SetBranchAddress("latpred",&latpred);
  tree->SetBranchAddress("lonpred",&lonpred);
  tree->SetBranchAddress("intrainFOV",&intrainFOV);

  float counter = 0;
  for(int i = 0; i < tree->GetEntries(); i++){
    tree->GetEntry(i);
    cout << lattrue << " " << latpred << " " << (lattrue-latpred)*(lattrue-latpred) << " || " <<  lontrue << " " << lonpred << " " << (lontrue-lonpred)*(lontrue-lonpred) << endl;
    histlat->Fill(lontrue,lattrue,(lattrue-latpred)*(lattrue-latpred));
    histlon->Fill(lontrue,lattrue,(lontrue-lonpred)*(lontrue-lonpred));
    histlatcount->Fill(lontrue,lattrue);
    histloncount->Fill(lonpred,latpred);
    histlatp->Fill(lonpred,latpred,(lattrue-latpred)*(lattrue-latpred));
    histlonp->Fill(lonpred,latpred,(lontrue-lonpred)*(lontrue-lonpred));
    if(intrainFOV){
      histlatzoom->Fill(lontrue,lattrue,(lattrue-latpred)*(lattrue-latpred));
      histlonzoom->Fill(lontrue,lattrue,(lontrue-lonpred)*(lontrue-lonpred));
      histlatcountzoom->Fill(lontrue,lattrue);
      histloncountzoom->Fill(lonpred,latpred);
      histlatpzoom->Fill(lonpred,latpred,(lattrue-latpred)*(lattrue-latpred));
      histlonpzoom->Fill(lonpred,latpred,(lontrue-lonpred)*(lontrue-lonpred));
      laterro += (lattrue-latpred)*(lattrue-latpred);
      lonerro += (lontrue-lonpred)*(lontrue-lonpred);
      //histlatconf->Fill(latpred,lattrue,(lattrue-latpred)*(lattrue-latpred));
      //histlonconf->Fill(lonpred,lontrue,(lontrue-lonpred)*(lontrue-lonpred));
      //histlatconf->Fill(latpred,lattrue);
      //histlonconf->Fill(lonpred,lontrue);
      glatconf->SetPoint(i,latpred,lattrue);
      glonconf->SetPoint(i,lonpred,lontrue);
      counter++;
    }else{
      glatconfOUT->SetPoint(i,latpred,lattrue);
      glonconfOUT->SetPoint(i,lonpred,lontrue);
    }
  }
  for(int i=1; i<=histlat->GetNbinsX(); i++){
    for(int j=1; j<=histlat->GetNbinsY(); j++){
      //if(histlat->GetBinContent(i,j) != 0)   cout<<histlat->GetBinContent(i,j)/ histlatcount->GetBinContent(i,j) << endl;
      if(histlat->GetBinContent(i,j) != 0) histlat->SetBinContent(i,j,histlat->GetBinContent(i,j)/histlatcount->GetBinContent(i,j));
      if(histlatp->GetBinContent(i,j) != 0) histlatp->SetBinContent(i,j,histlatp->GetBinContent(i,j)/histloncount->GetBinContent(i,j));
      if(histlon->GetBinContent(i,j) != 0) histlon->SetBinContent(i,j,histlon->GetBinContent(i,j)/histlatcount->GetBinContent(i,j));
      if(histlonp->GetBinContent(i,j) != 0) histlonp->SetBinContent(i,j,histlonp->GetBinContent(i,j)/histloncount->GetBinContent(i,j));
      if(histlatzoom->GetBinContent(i,j) != 0) histlatzoom->SetBinContent(i,j,histlatzoom->GetBinContent(i,j)/histlatcountzoom->GetBinContent(i,j));
      if(histlatpzoom->GetBinContent(i,j) != 0) histlatpzoom->SetBinContent(i,j,histlatpzoom->GetBinContent(i,j)/histloncountzoom->GetBinContent(i,j));
      if(histlonzoom->GetBinContent(i,j) != 0) histlonzoom->SetBinContent(i,j,histlonzoom->GetBinContent(i,j)/histlatcountzoom->GetBinContent(i,j));
      if(histlonpzoom->GetBinContent(i,j) != 0) histlonpzoom->SetBinContent(i,j,histlonpzoom->GetBinContent(i,j)/histloncountzoom->GetBinContent(i,j));
    }
  }
  laterro = laterro/counter;
  lonerro = lonerro/counter;

  gStyle->SetPalette(54);
  TLine llat(-38,-38,-30,-30);
  TLine llon(-68,-68,-60,-60);

  TCanvas c1("c1","c1",1000,1000);

  gStyle->SetOptStat(0);
  histlat->Draw("colz");
  c1.SaveAs("laterror.png");
  histlon->Draw("colz");
  c1.SaveAs("lonerror.png");
  histlatp->Draw("colz");
  c1.SaveAs("laterrorp.png");
  histlonp->Draw("colz");
  c1.SaveAs("lonerrorp.png");

  histlatzoom->Draw("colz");
  c1.SaveAs("laterrorzoom.png");
  histlonzoom->Draw("colz");
  c1.SaveAs("lonerrorzoom.png");
  histlatpzoom->Draw("colz");
  c1.SaveAs("laterrorpzoom.png");
  histlonpzoom->Draw("colz");
  c1.SaveAs("lonerrorpzoom.png");



  c1.SetLeftMargin(1.8);
  histlatconf->Draw();
  histlatconf->GetYaxis()->SetTitleOffset(1.85);
  histlatconf->SetTitle(TString::Format("Average MSE (w/in range) = %1.2f",laterro));
  histlatconf->GetYaxis()->SetLabelSize(0.026);
  histlatconf->GetXaxis()->SetLabelSize(0.026);
  histlatconf->GetXaxis()->SetTitleSize(0.026);
  histlatconf->GetYaxis()->SetTitleSize(0.026);
  glatconf->SetMarkerSize(1.5);
  glatconf->SetMarkerStyle(20);
  glatconf->SetMarkerColor(kBlue);
  glatconf->Draw("P");
  glatconfOUT->SetMarkerSize(1.5);
  glatconfOUT->SetMarkerStyle(20);
  glatconfOUT->SetMarkerColor(kRed);
  glatconfOUT->Draw("P");
  llat.SetLineWidth(2.0);
  llat.SetLineColor(kBlack);
  llat.Draw("");
  c1.SaveAs("laterrorconf.png");

  c1.SetLeftMargin(1.8);
  histlonconf->Draw();
  histlonconf->GetYaxis()->SetTitleOffset(1.85);
  histlonconf->SetTitle(TString::Format("Average MSE (w/in range) = %1.2f",lonerro));
  histlonconf->GetYaxis()->SetLabelSize(0.026);
  histlonconf->GetXaxis()->SetLabelSize(0.026);
  histlonconf->GetXaxis()->SetTitleSize(0.026);
  histlonconf->GetYaxis()->SetTitleSize(0.026);
  glonconf->SetMarkerSize(1.5);
  glonconf->SetMarkerStyle(20);
  glonconf->SetMarkerColor(kBlue);
  glonconf->Draw("P");
  glonconfOUT->SetMarkerSize(1.5);
  glonconfOUT->SetMarkerStyle(20);
  glonconfOUT->SetMarkerColor(kRed);
  glonconfOUT->Draw("P");
  llon.SetLineWidth(2.0);
  llon.SetLineColor(kBlack);
  llon.Draw("");
  c1.SaveAs("lonerrorconf.png");

  const int NRGBs = 5;
  const int MaxColours = 127;//roberto changed this to 127 in reconstruction
  int palette[MaxColours];
  Double_t stops[NRGBs] = { 0.00, 0.34, 0.61, 0.84, 1.00 };
  Double_t red[NRGBs]   = { 0.00, 0.00, 0.87, 1.00, 0.51 };
  Double_t green[NRGBs] = { 0.00, 0.81, 1.00, 0.20, 0.00 };
  Double_t blue[NRGBs]  = { 0.51, 1.00, 0.12, 0.00, 0.00 };
  Int_t FI = TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, MaxColours);
  for (int i=0;i<MaxColours;i++) {
    palette[i] = FI+i;
    //    cout << palette[i] << endl;
  }
  TColor::SetPalette(MaxColours,palette);
  gStyle->SetNumberContours(MaxColours);//only do the analysis if dealing with the run of choice.

  //fhIntegralPixelPalette[j] = (int)MaxColours * (fhIntegralPixel->GetBinContent(j+1) - 1.0*fhIntegralPixel->GetMinimum())/(1.1*fhIntegralPixel->GetMaximum()- 1.0*fhIntegralPixel->GetMinimum()) + palette[0] ;
  TH1F laterror("laterror","",tree->GetEntries(),1,tree->GetEntries()); 
  TH1F lonerror("lonerror","",tree->GetEntries(),1,tree->GetEntries()); 
  const int nentries = (const int)tree->GetEntries();
  cout << "HERE" << endl;
  int laterrorPalette[nentries];
  int lonerrorPalette[nentries];
  int FOVflag = 1;
  for(int i = 0; i < nentries; i++){
    tree->GetEntry(i);
    if(!FOVflag) intrainFOV=1;
    if(intrainFOV){
      laterror.Fill(i+1,(lattrue-latpred)*(lattrue-latpred));
      lonerror.Fill(i+1,(lontrue-lonpred)*(lontrue-lonpred));
    }
  }

  TGraph* glaterror[nentries];
  TGraph* glonerror[nentries];
  TGraph* glaterrorp[nentries];
  TGraph* glonerrorp[nentries];
  for(int i = 0; i < nentries; i++){
    tree->GetEntry(i);
    laterrorPalette[i] = (int)MaxColours * (laterror.GetBinContent(i+1) - 1.0*laterror.GetMinimum()) / (1.0*laterror.GetMaximum()- 1.0*laterror.GetMinimum())+ palette[0];
    lonerrorPalette[i] = (int)MaxColours * (lonerror.GetBinContent(i+1) - 1.0*lonerror.GetMinimum()) / (1.0*lonerror.GetMaximum()- 1.0*lonerror.GetMinimum())+ palette[0];
    cout << intrainFOV << " " << lattrue << " " << lontrue << " "  << laterrorPalette[i] << " " << lonerrorPalette[i] <<  " " << laterror.GetBinContent(i+1) << " " << lonerror.GetBinContent(i+1) << endl;

    glaterror[i] = new TGraph(1);
    glonerror[i] = new TGraph(1);
    if(!FOVflag) intrainFOV=1;
    if(intrainFOV){
      glaterror[i]->SetPoint(i,lontrue,lattrue);
      glaterror[i]->SetMarkerColor(laterrorPalette[i]);
      glaterror[i]->SetMarkerSize(2.0);
      glaterror[i]->SetMarkerStyle(20);
      
      glonerror[i]->SetPoint(i,lontrue,lattrue);
      glonerror[i]->SetMarkerColor(lonerrorPalette[i]);
      glonerror[i]->SetMarkerSize(2.0);
      glonerror[i]->SetMarkerStyle(20);
    }
    glaterrorp[i] = new TGraph(1);
    glonerrorp[i] = new TGraph(1);
	     
    if(intrainFOV){
      glaterrorp[i]->SetPoint(i,lonpred,latpred);
      glaterrorp[i]->SetMarkerColor(laterrorPalette[i]);
      glaterrorp[i]->SetMarkerSize(2.0);
      glaterrorp[i]->SetMarkerStyle(20);
      
      glonerrorp[i]->SetPoint(i,lonpred,latpred);
      glonerrorp[i]->SetMarkerColor(lonerrorPalette[i]);
      glonerrorp[i]->SetMarkerSize(2.0);
      glonerrorp[i]->SetMarkerStyle(20);
    }
  }

  
  TH2F * histlatitude = new TH2F("histlatitude","MSE LATITUDE TRUTH MAP;True Longitude;True Latitude;MSE",40,-68,-60,nentries,-38,-30);
  histlatitude->GetYaxis()->SetTitleOffset(1.85);
  histlatitude->GetYaxis()->SetLabelSize(0.026);
  histlatitude->GetXaxis()->SetLabelSize(0.026);
  histlatitude->GetXaxis()->SetTitleSize(0.026);
  histlatitude->GetYaxis()->SetTitleSize(0.026);
  for (int i=0;i<nentries;i++) {
    histlatitude->SetBinContent(0,i+1,laterror.GetBinContent(i+1));
  }
  histlatitude->GetZaxis()->SetRangeUser(laterror.GetMinimum(),laterror.GetMaximum());
  histlatitude->Draw("colz");
  gPad->Update();
  TPaletteAxis *paletteaxis = (TPaletteAxis*)histlatitude->GetListOfFunctions()->FindObject("palette");
  paletteaxis->SetX1NDC(0.90);
  paletteaxis->SetY1NDC(0.1);
  paletteaxis->SetX2NDC(0.925);
  paletteaxis->SetY2NDC(0.9);
  for(int i = 0; i < nentries; i++){
    glaterror[i]->Draw("P");
  }
  c1.SaveAs("mselattruth.png");
  histlatitude->SetTitle("MSE LATITUDE PREDICTED MAP;Predicted Longitude;Predicted Latitude;MSE");
  histlatitude->Draw("colz");
  for(int i = 0; i < nentries; i++){
    glaterrorp[i]->Draw("P");
  }
  c1.SaveAs("mselatpred.png");
  histlatitude->SetTitle("MSE LONGITUDE TRUTH MAP;True Longitude;True Latitude;MSE");
  for (int i=0;i<nentries;i++) {
    histlatitude->SetBinContent(0,i+1,lonerror.GetBinContent(i+1));
  }
  histlatitude->GetZaxis()->SetRangeUser(lonerror.GetMinimum(),lonerror.GetMaximum());
  histlatitude->Draw("colz");
  for(int i = 0; i < nentries; i++){
    glonerror[i]->Draw("P");
  }
  c1.SaveAs("mselontruth.png");
  histlatitude->SetTitle("MSE LONGITUDE PREDICTED MAP;Predicted Longitude;Predicted Latitude;MSE");
  histlatitude->Draw("colz");
  for(int i = 0; i < nentries; i++){
    glonerrorp[i]->Draw("P");
  }
  c1.SaveAs("mselonpred.png");

}
