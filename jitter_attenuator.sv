`timescale 1ns/1ps
module jiiter_attenuator(input_clk, N,output_clk);
input int N;
input input_clk;
output output_clk;
reg rst,frac;
reg input_clk_internal;
wire[6:0] DLF_IN;
wire signed[26:0] DLF_out;
wire signed[26:0] QNC ;
//reg signed[23:0] alpha1 ;

//reg signed[23:0] alpha;

TDC #(.RESOLUTION (10e-12), .DOUT_WIDTH(7)) tdc (.Clk_ref(input_clk),.feedback(ClK_Div),.Dout(DLF_IN),.time_diff());
//TDC #(.RESOLUTION (10e-12), .DOUT_WIDTH(7) , .time_unit(1e-15)) tdc (.Clk_ref(input_clk),.feedback(ClK_Div),.Dout(DLF_IN)); //New comment
DLF #(.alpha(596.3130),.beta(1.420953e-1),.WIDTH_IN(7),.WIDTH_OUT(27)) dlf (.clk(input_clk),.rst(rst),.IN(DLF_IN),.OUT(DLF_out));
sysDiv internalpll(input_clk_internal,N,output_clk,DLF_out);
clkdiveven #(.N(90))divider(.vco_out(output_clk),.clk_out(ClK_Div),.x(frac));
Sigma_Delta#(.n(26))sigmadelta(.alpha(27'd12),.CLK(ClK_Div) ,.out(frac),.QNC(QNC));



always
#3.025 input_clk_internal=~input_clk_internal;
initial begin rst=1;#0.01 rst=0; input_clk_internal=0; end
endmodule