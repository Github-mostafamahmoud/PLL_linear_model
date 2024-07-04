`timescale 1ns/1fs
module VCO_tb_new();
// Declare signals
real Fref,min_freq,max_freq,kvco,Vcont,Voutp,Voutn,freq_measured;
reg oclk;
real tt2; //variable to save realtime of rising edge of output
real tt3; //variable to save realtime of falling edge of output
real tt4; //variable represent period of output
// Instantiate the module under test
VCO_new #(.vin_max(1.2),.kvco(55e6),.min_freq(9e9))dut(.oclk(oclk),.Voutp(Voutp),.Voutn(Voutn),.cfs(6'd17));
initial begin
Voutp=0.294545;
Voutn=-0.294545;
#0.3;
Voutp=0;
Voutn=-0;
#0.3;
Voutp=0.6;
Voutn=-0.6;
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