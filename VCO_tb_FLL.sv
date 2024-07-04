`timescale 1ns/1ps
module VCO_tb_FLL();

// Declare signals
real Fref,min_freq,max_freq,kvco,Vmax;
integer N;
wire unsigned [5:0] cfs;
// Instantiate the module under test
VCO_FLL #(.Fref(150e6),.Vmax(1.2),.kvco(55e6),.min_freq(9e9),.max_freq(11e9))dut(.N(N),.cfs(cfs));
initial begin
N=67;
end
endmodule