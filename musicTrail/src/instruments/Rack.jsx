import AltScope from './AltScope.jsx'
import GradeLadder from './GradeLadder.jsx'
import BearingRose from './BearingRose.jsx'
import CadenceTrace from './CadenceTrace.jsx'
import GrooveComb from './GrooveComb.jsx'
import WindVector from './WindVector.jsx'
import BarsRuler from './BarsRuler.jsx'
import LockStrip from './LockStrip.jsx'

// A shared instrument well: hairline bracket + a name. No panel fill, no bezel.
export function Mod({ name, sub, className = '', children }) {
  return (
    <section className={`mod ${className}`}>
      <div className="mod__h">
        <span className="w">{name}</span>
        {sub && <span className="n n--10 n--lo">{sub}</span>}
      </div>
      {children}
    </section>
  )
}

// The instrument stack: everything the rig is measuring about this climb.
export default function Rack({ channel, camp, target, route }) {
  const tgt = target && target.id !== camp.id ? target : null
  return (
    <aside className="rack">
      <AltScope camp={camp} route={route} />
      <BearingRose route={route} />
      <CadenceTrace camp={camp} route={route} />
      <GradeLadder route={route} />
      <BarsRuler route={route} />
      <WindVector route={route} />
      <GrooveComb route={route} target={tgt} />
      <LockStrip channel={channel} route={route} />
    </aside>
  )
}
