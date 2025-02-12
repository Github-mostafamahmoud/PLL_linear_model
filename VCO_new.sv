`timescale 1ns/1fs

/////////////////////////////////////////////////////

module VCO_new(oclk,N,Voutp,Voutn,cfs);
input integer N;
//parameter real vin_max=1.2;
parameter real min_freq=9e9;
parameter real max_freq=11e9;
input [5:0] cfs;
parameter real kvco=55e6;
//parameter real max_analog_outp,max_analog_outn;
parameter real Vmax;
input real Voutp,Voutn;
real fcont,freq_out,Tout,lock_freq,last_min,temp;//Vmax;//Vmid,Vmax;
//integer Ncurves,i;
real vin_min=0;
real Vcont;
reg clk;
real tt1; //half period of output
output oclk;
assign Vcont=Voutp-Voutn;
real step= 0.0378e9;//33e6;//0.0378e9; /// .033e9 overlab 50%
real temp_freq;

always@(*)
begin
temp_freq=min_freq+(cfs*step);
if(Vcont<Vmax && Vcont>vin_min) begin
	fcont=Vcont*kvco;
end
else if(Vcont>=Vmax) begin
	fcont=Vmax*kvco;
end	
else if(Vcont<=vin_min)begin
	fcont=0;
 
end
 
freq_out=fcont+temp_freq;
Tout=1/freq_out;
tt1=0.5*Tout*1e9;
end

initial begin
temp_freq=min_freq;

end 
always begin
    #tt1; clk=~clk;end
initial begin
    clk = 0;
    temp_freq=min_freq;
    fcont=1;
end
assign oclk=clk;
 
endmodule
