import { app } from "../../../../scripts/app.js";

export class FlowState {
    constructor(){}
    static idle() {
        return (!app.runningNodeId);
    }
    static paused() {
        return true;
    }
    static paused_here(node_id) {
        return (FlowState.paused() && FlowState.here(node_id))
    }
    static running() {
        return (!FlowState.idle());
    }
    static here(node_id) {
        return (app.runningNodeId==node_id);
    }
    static state() {
        if (FlowState.paused()) return "Paused";
        if (FlowState.running()) return "Running";
        return "Idle";
    }
    static cancelling = false;
}