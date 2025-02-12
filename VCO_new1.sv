`timescale 1ns/1fs
module VCO_new1(oclk,Voutp,Voutn,cfs);
input real Voutp,Voutn;
input unsigned [5:0] cfs;
output oclk;

parameter real vin_max,kvco,min_freq;

real fcont,freq_out,Tout,temp_freq;
real tt1; //half period of output
real vin_min=0;
real Vcont;
reg clk;


assign Vcont=Voutp-Voutn;

//define step 
real step=0.0378e9;
assign temp_freq=min_freq+(cfs*step);

always@(*)
begin

if(Vcont<vin_max && Vcont>vin_min) begin
	fcont=Vcont*kvco;
end
else if(Vcont>=vin_max) begin
	fcont=vin_max*kvco;
end	
else if(Vcont<=vin_min)begin
	fcont=0;
end
 freq_out=fcont+temp_freq;
 Tout=1/freq_out;
 tt1=0.5*Tout*1e9;

end


always
    #tt1 clk=~clk;
initial 
    clk = 0;
assign oclk=clk;

endmodule

