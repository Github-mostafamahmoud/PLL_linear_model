.main clear
quit -sim

cd {D:/college/senior 2/Graduation Project/Jitter_system/PLL__Modeling-main_new2}

vlog jitter_attenuator.sv
vlog jitter_attenuator_tb.sv
vsim jiiter_attenuator_tb -wlf ./vsim.wlf -L work
log -r jiiter_attenuator_tb/*

add wave -position insertpoint \
sim:/jiiter_attenuator_tb/attenuator/internalpll/vco/Vcont \
sim:/jiiter_attenuator_tb/attenuator/internalpll/vco/freq_out \
sim:/jiiter_attenuator_tb/attenuator/internalpll/vco/temp_freq \
sim:/jiiter_attenuator_tb/attenuator/internalpll/vco/cfs \
sim:/jiiter_attenuator_tb/attenuator/internalpll/tdc/Dout \
sim:/jiiter_attenuator_tb/attenuator/internalpll/tdc/time_diff \
sim:/jiiter_attenuator_tb/attenuator/internalpll/dlf/OUT \
sim:/jiiter_attenuator_tb/attenuator/internalpll/sigmadelta/out \
sim:/jiiter_attenuator_tb/attenuator/internalpll/sigmadelta/division \
sim:/jiiter_attenuator_tb/attenuator/internalpll/divider/Division_actual \
sim:/jiiter_attenuator_tb/attenuator/tdc/Dout \
sim:/jiiter_attenuator_tb/attenuator/tdc/time_diff \
sim:/jiiter_attenuator_tb/attenuator/sigmadelta/out \
sim:/jiiter_attenuator_tb/attenuator/dlf/OUT

run 2ms