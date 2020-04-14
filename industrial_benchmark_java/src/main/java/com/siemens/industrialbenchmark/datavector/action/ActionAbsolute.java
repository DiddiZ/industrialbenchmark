/**
Copyright 2016 Siemens AG.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package com.siemens.industrialbenchmark.datavector.action;

import java.util.Properties;

import com.google.common.base.Preconditions;
import com.siemens.industrialbenchmark.datavector.state.ObservableStateDescription;
import com.siemens.industrialbenchmark.properties.PropertiesException;
import com.siemens.industrialbenchmark.properties.PropertiesUtil;

/**
 * This class keeps and checks for the actions velocity, gain and shift. Internally it is based on 
 * DELTA actions, which are computed on basis of the last values.
 *  
 * @author Michel Tokic
 */
public class ActionAbsolute extends ActionDelta {

	/**
	 * 
	 */
	private static final long serialVersionUID = 802570663649527391L;
	
	private double  absVelocity = 0;
	private double  absGain = 0;
	private double  absShift = 0;
	
	private final double  velocityMin;
	private final double  velocityMax;
	private final double  gainMin;
	private final double  gainMax;
	private final double  shiftMin;
	private final double  shiftMax;
	
	/**
	 * Constructor actions and properties file
	 * @param velocity The velocity to set 
	 * @param gain The gain to set
	 * @param shift The shift to set
	 * @param props The Properties file with boundaries for velocity, gain and shift 
	 * @throws PropertiesException
	 */
	public ActionAbsolute (double velocity, double gain, double shift, Properties props) throws PropertiesException {
		
		super(0, 0, 0);

		this.velocityMin = PropertiesUtil.getFloat(props, ObservableStateDescription.Action_Velocity + "_MIN", 0f);
		this.velocityMax = PropertiesUtil.getFloat(props, ObservableStateDescription.Action_Velocity + "_MAX", 100f);
		this.gainMin = PropertiesUtil.getFloat(props, ObservableStateDescription.Action_Gain + "_MIN", 0f);
		this.gainMax = PropertiesUtil.getFloat(props, ObservableStateDescription.Action_Gain + "_MAX", 100f);
		this.shiftMin = PropertiesUtil.getFloat(props, ObservableStateDescription.Action_Shift + "_MIN", 0f);
		this.shiftMax = PropertiesUtil.getFloat(props, ObservableStateDescription.Action_Shift + "_MAX", 100f);
			
		Preconditions.checkArgument(velocity >= velocityMin && velocity <= velocityMax, "velocity=%s must be in range [%s, %s]", velocity, velocityMin, velocityMax);
		Preconditions.checkArgument(gain >= gainMin && gain <= gainMax, "gain=%s must be in range [%s, %s]", gain, gainMin, gainMax);
		Preconditions.checkArgument(shift >= gainMin && shift <= gainMax, "shift=%s must be in range [%s, %s]", shift, shiftMin, shiftMax);

		this.absVelocity = velocity;
		this.absGain = gain;
		this.absShift = shift;
	}

	/**
	 * @return the velocity
	 */
	public double  getVelocity() {
		return absVelocity;
	}

	/**
	 * @return the gain
	 */
	public double  getGain() {
		return absGain;
	}
	
	/**
	 * @return the shift
	 */
	public double getShift() {
		return absShift;
	}

	/**
	 * @param velocity the A to set
	 */
	public void setVelocity(double velocity) {
		double  delta = Math.abs(velocity - absVelocity);
		Preconditions.checkArgument(velocity >= velocityMin && velocity <= velocityMax, "velocity=%s must be in range [%s, %s].", velocity, velocityMin, velocityMax);
		Preconditions.checkArgument(delta <= maxDelta, "delta_velocity=%s out of range. 'Velocity' must be in range [%s, %s].", absVelocity-delta, absVelocity+delta);
		this.setValue(ActionDeltaDescription.DeltaVelocity, velocity - this.absVelocity);
		this.absVelocity = velocity;
	}

	/**
	 * @param gain the gain to set
	 */
	public void setGain(double gain) {
		double  delta = Math.abs(gain- absGain);
		Preconditions.checkArgument(gain>= gainMin && gain <= gainMax, "gain=%s must be in range [%s, %s].", gain, gainMin, gainMax);
		Preconditions.checkArgument(delta <= maxDelta, "delta_gain=%s out of range. 'gain' must be in range [%s, %s].", absGain-delta, absGain+delta);  
		this.setValue(ActionDeltaDescription.DeltaGain, gain - this.absGain);
		this.absGain = gain;
	}
	
	/**
	 * @param shift the shift to set
	 */
	public void setShift(float shift) {
		double delta = Math.abs(shift- absShift);
		Preconditions.checkArgument(shift>= shiftMin && shift <= shiftMax, "=%s must be in range [%s, %s].", shift, shiftMin, shiftMax);
		Preconditions.checkArgument(delta <= maxDelta, "delta_shift=%s out of range. 'C' must be in range [%s, %s].", absShift-delta, absShift+delta);  
		this.setValue(ActionDeltaDescription.DeltaShift, shift - this.absShift);
		this.absShift = shift;
	}
}
