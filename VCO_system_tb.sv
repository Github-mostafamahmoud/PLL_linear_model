`timescale 1ns/1fs
module VCO_system_tb();
real Fref,min_freq,max_freq,kvco,Vcont,Voutp,Voutn,freq_measured;
integer N;
reg oclk;
real tt2; //variable to save realtime of rising edge of output
real tt3; //variable to save realtime of falling edge of output
real tt4; //variable represent period of output
// Instantiate the module under test
VCO_system #(.Fref(150e6),.Vmax(1.2),.kvco(55e6),.min_freq(9e9),.max_freq(11e9))dut(.N(N),.oclk(oclk),.Voutp(Voutp),.Voutn(Voutn));
initial begin
N=64;
Voutp=0.3;
Voutn=-0.3;
#1;
N=66;
#1;
N=67;
end

always@(posedge oclk or negedge oclk) begin
if(oclk)
	tt2=$realtime();
else if(~oclk)
	tt3=$realtime();
 tt4=(tt2-tt3);
 if(tt4<0)
 tt4=-tt4;
freq_measured=1.0/(2*tt4);

if(tt4==0)
freq_measured=0;

end
endmodule