`timescale 1ns/1fs
module VCO_system(oclk,Voutp,Voutn,N,alpha);
input real Voutp,Voutn;
input integer N;

output oclk;

parameter real Fref,min_freq,max_freq,kvco,Vmax;
parameter integer n;
wire unsigned [5:0] cfs;

input signed [n:0] alpha;

VCO_FLL #(.n(n),.Fref(Fref),.Vmax(Vmax),.kvco(kvco),.min_freq(min_freq),.max_freq(max_freq))dut(.N(N),.cfs(cfs),.alpha(alpha));
VCO_new1 #(.vin_max(Vmax),.kvco(kvco),.min_freq(min_freq))dut1(.oclk(oclk),.Voutp(Voutp),.Voutn(Voutn),.cfs(cfs));
endmodule 