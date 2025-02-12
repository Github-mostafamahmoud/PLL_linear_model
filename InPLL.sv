`timescale 1ns/1ps
module InPLL (DLF_Cont,output_clk);
parameter integer n;
 real Fref= 165.289256e6;

input [n:0] DLF_Cont;
output output_clk;
reg rst,error,input_clk;
reg signed[n:0] alpha ;
reg signed[n:0] alpha_set;
reg frac;
wire[6:0] DLF_IN;
wire[17:0] DAC_IN;
wire signed [n:0] QNC ;
wire ClK_Div ,DTC_OUT;

real analog_outp,analog_outn,Voutp_max,Voutn_max,fvco,tvco_out;

reg signed [9:0] QNC_CORR_10bit;


TDC #(.RESOLUTION (1e-13), .DOUT_WIDTH(7)) tdc (.Clk_ref(input_clk),.feedback(DTC_OUT),.Dout(DLF_IN),.time_diff());
DLF #(.WIDTH_IN(7),.WIDTH_OUT(18)) dlf (.clk(input_clk),.rst(rst),.IN(DLF_IN),.OUT(DAC_IN));
DAC #(.DATA_WIDTH(18)) dac (.reset(rst),.data(DAC_IN),.analog_outp(analog_outp),.analog_outn(analog_outn),
.max_analog_outp(Voutp_max),.max_analog_outn(Voutn_max));

VCO_new #(.Vmax(1.2),.kvco(55e6),.min_freq(9e9),.max_freq(11e9))vco(.oclk(output_clk),.cfs(6'd25),.Voutp(analog_outp)
,.Voutn(analog_outn));
clkdiveven #(.N(60))divider(.vco_out(output_clk),.clk_out(ClK_Div),.x(frac));
Sigma_Delta_internal#(.n(n))sigmadelta(.alpha(alpha), .CLK(ClK_Div) , .out(frac),.QNC(QNC) );
TDC #(.RESOLUTION (1e-13), .DOUT_WIDTH(7)) freq_mesure (.Clk_ref(output_clk),.feedback(~output_clk),.time_diff(tvco_out),.Dout());
assign fvco = 1.0/(2*tvco_out);

always 
#3.025 input_clk=~input_clk;

always @(posedge input_clk or posedge DTC_OUT)
begin
    if ((input_clk && !ClK_Div) || (!input_clk && ClK_Div))
        error = input_clk - ClK_Div;
      else error=0;
end

assign QNC_CORR_10bit= (QNC/(2**(n-5))-8);

DTC #(.DIN_WIDTH(10))dtc (
 .d_in(QNC_CORR_10bit), // Digital input to DTC
 .sig_in(ClK_Div),   // Input Signal
 .sig_out(DTC_OUT) //  delayed output signal
);
assign alpha=alpha_set+DLF_Cont;
initial begin rst=1;#0.01 rst=0; alpha_set=2**(n-2);input_clk=0; 
end
endmodule

